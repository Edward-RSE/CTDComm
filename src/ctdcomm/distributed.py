import os
import socket

import torch
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist


from ctdcomm.utils import merge_stat
from ctdcomm.trainer import Trainer

MAX_MESSAGE_SIZE = 32


class DistributedWorker:
    def __init__(
        self,
        global_rank,
        global_world_size,
        global_args,
        trainer,
        global_seed,
        *,
        save_adjacency=False,
    ):
        self.rank = global_rank
        self.world_size = global_world_size
        self.local_rank = self.local_world_size = -1
        self.dist_backend = "nccl" if global_args.use_cuda else "gloo"
        self.port = self.address = None
        self._init_distributed_ranks(
            self.rank, self.global_world_size, self.dist_backend
        )
        self._initialise_local_ranks()

        self.args = global_args
        self.seed = global_seed + global_rank
        self.trainer: Trainer = trainer  # TODO(EP): remove type hinting later
        self.save_adjacency = save_adjacency
        self.device = None

        # Initialise devices and then create a distribute model, to replace the
        # original policy net, and push that to the appropriate devices
        self._initialise_devices()
        self.trainer.wrap_with_distributed_policy_net(self.device)

    @staticmethod
    def _find_available_socket(hostname):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((hostname, 0))  # Bind to the first available port
            return s.getsockname()[1]

    def _init_distributed_ranks(self, global_rank, global_world_size, backend):
        # If torchrun is used, then we don't need to set these env vars so use
        # setdefault to use torchrun values otherwise we set localhost etc
        os.environ.setdefault("MASTER_ADDR", "localhost")
        self.address = os.environ["MASTER_ADDR"]
        os.environ.setdefault(
            "MASTER_PORT", str(self._find_available_socket(self.address))
        )
        self.port = os.environ["MASTER_PORT"]
        torch.distributed.init_process_group(
            backend,
            rank=global_rank,
            world_size=global_world_size,
        )
        self.rank = dist.get_rank()
        print(f"Created global rank {self.rank} on {socket.gethostname()}")

    def _destroy_distributed_ranks(self):
        print(f"Destroying global rank {self.rank} on {socket.gethostname()}")
        dist.destroy_process_group()

    def _initialise_local_ranks(self):
        # torchrun will set LOCAL_RANK and LOCAL_WORLD_SIZE for us. If these are
        # not set, then it is safe to assume that we are probably on the same
        # node so we can re-use the global values
        if os.environ["LOCAL_RANK"]:
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        if os.environ["LOCAL_WORLD_SIZE"]:
            self.local_world_size = int(
                os.environ.get("LOCAL_WORLD_SIZE", self.world_size)
            )
        print(
            f"Created local rank {self.local_rank}/{self.local_world_size} on {socket.gethostname()}"
        )

    def _initialise_devices(self):
        hostname = socket.gethostname()
        if self.args.use_cuda and torch.cuda.is_available():
            if self.local_world_size > torch.cuda.device_count():
                raise RuntimeError(
                    f"Local world size of {self.local_world_size} on {hostname} is greater than device count of {torch.cuda.device_count()}"
                )
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        print(f"Global rank {self.rank} is using device {self.device} on {hostname}")

    def run(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        for epoch in range(self.args.num_epochs):
            if self.save_adjacency:
                batch, stat, adjacency_data = self.trainer.train_batch(epoch)
            else:
                batch, stat = self.trainer.train_batch(epoch)
            self.trainer.optimizer.zero_grad()
            self.trainer.compute_grad(batch)
            # merge_stat(s, stat)

        # while True:
        #     task = self.comm.recv()
        #     if type(task) == list:
        #         task, epoch = task

        #     if task == "quit":
        #         self.trainer.env.close()
        #         return
        #     elif task == "run_batch":
        #         if self.save_adjacency:
        #             batch, stat, batch_adjacency = self.trainer.run_batch(epoch)
        #         else:
        #             batch, stat = self.trainer.run_batch(epoch)
        #         self.trainer.optimizer.zero_grad()
        #         s = self.trainer.compute_grad(batch)
        #         merge_stat(s, stat)
        #         if self.save_adjacency:
        #             self.comm.send([stat, batch_adjacency])
        #         else:
        #             self.comm.send(stat)
        #     elif task == "send_grads":
        #         grads = []
        #         for p in self.trainer.params:
        #             if p._grad is not None:
        #                 grads.append(p._grad.data)

        #         self.comm.send(grads)


class DistributedTrainer(object):
    def __init__(self, args, trainer_maker):
        self.args = args
        self.world_size = args.nprocesses
        self.trainer = trainer_maker()

    def run(self):
        mp.spawn(
            self._worker_entry,
            args=(self.world_size,),
            nprocs=self.world_size,
            join=True,
        )

    def _worker_entry(self, global_rank, global_world_size):
        worker = DistributedWorker(
            global_rank,
            global_world_size,
            self.args,
            self.trainer,
            self.seed,
            save_adjacency=self.args.save_adjacency,
        )
        worker.run()

    def train_batch(self, epoch):
        for rank in range(self.world_size):
            dist.send_object_list(["run_batch", epoch])
