import os

import numpy as np
import torch
import torch.distributed as dist

from ctdcomm.trainer import Trainer
from ctdcomm.utils import display_models, merge_stat

ROOT_RANK = 0


class DistributedTrainer:
    def __init__(
        self,
        args,
        trainer_maker,
        *,
        save_adjacency=False,
    ):
        self.root_rank = ROOT_RANK
        self.rank = self.world_size = -1
        self.local_rank = self.local_world_size = -1
        self.dist_backend = "nccl" if args.use_cuda else "gloo"
        self.port = self.address = self.hostname = None
        self._initialise_local_ranks()
        self._init_distributed_ranks(self.rank, self.world_size, self.dist_backend)

        self.args = args
        self.seed = args.seed + self.rank + 1
        self.trainer: Trainer = trainer_maker()  # TODO(EP): remove type hinting later
        self.save_adjacency = save_adjacency
        self.device = None

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialise devices and then create a distribute model, to replace the
        # original policy net, and push that to the appropriate devices
        self._initialise_devices()
        # self.trainer.wrap_with_distributed_policy_net(self.device)

        print(
            f"Global rank {self.rank} is using device {self.device} on {self.address}:{self.port}"
        )

        if self.rank == 0:
            print(self.args)
            display_models([self.trainer.policy_net])
        dist.barrier()

    def _init_distributed_ranks(self, rank, world_size, backend):
        # If torchrun is used, then we don't need to set these env vars so use
        # setdefault to use torchrun values otherwise we set localhost etc
        self.address = os.environ.setdefault("MASTER_ADDR", "localhost")
        self.port = os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(
            backend,
            rank=rank,
            world_size=world_size,
        )
        self.rank = dist.get_rank()

    def _destroy_distributed_ranks(self):
        print(f"Destroying global rank {self.rank} on {self.address}:{self.port}")
        dist.destroy_process_group()

    def _initialise_local_ranks(self):
        # torchrun will set RANK and WORLD_SIZE for us. If these are
        # not set, then it is safe to assume that we are probably on the same
        # node so we can re-use the global values
        self.rank = int(os.environ.setdefault("RANK", str(0)))
        self.world_size = int(os.environ.setdefault("WORLD_SIZE", str(1)))
        # torchrun will also set LOCAL_RANK and LOCAL_WORLD_SIZE for us.
        self.local_rank = int(os.environ.setdefault("LOCAL_RANK", str(self.rank)))
        self.local_world_size = int(
            os.environ.setdefault("LOCAL_WORLD_SIZE", str(self.world_size))
        )

    def _initialise_devices(self):
        if self.args.use_cuda and torch.cuda.is_available():
            if self.local_world_size > torch.cuda.device_count():
                raise RuntimeError(
                    f"Local world size of {self.local_world_size} on {self.address}:{self.port} is greater than device count of {torch.cuda.device_count()}"
                )
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

    def quit(self):
        self.trainer.env.close()
        self._destroy_distributed_ranks()

    # def obtain_grad_pointers(self):
    #     if self.grads is None:
    #         self.grads = []
    #         for p in self.trainer.params:
    #             if p._grad is not None:
    #                 self.grads.append(p._grad.data)

    def _gather_stat(self, stat):
        gathered_stat = (
            [None] * self.world_size if self.rank == self.root_rank else None
        )
        dist.gather_object(stat, gathered_stat, dst=self.root_rank)

        if self.rank == self.root_rank:
            for s in gathered_stat:
                merge_stat(s, stat)

        return stat

    def _gather_adjacency(self, adjacency):
        gathered_adjacency = (
            [None] * self.world_size if self.rank == self.root_rank else None
        )
        dist.gather_object(adjacency, gathered_adjacency, dst=self.root_rank)

        if self.rank == self.root_rank:
            for a in gathered_adjacency:
                adjacency += a

        return adjacency

    def train_batch(self, epoch):
        if self.save_adjacency:
            batch, stat, batch_adjacency = self.trainer.run_batch(epoch)
            mp_adjacency = list(batch_adjacency)
        else:
            batch, stat = self.trainer.run_batch(epoch)
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(batch)
        self.trainer.optimizer.step()
        merge_stat(s, stat)

        stat = self._gather_stat(stat)
        if self.save_adjacency:
            mp_adjacency = self._gather_adjacency(mp_adjacency)

        # add gradients of workers
        # self.obtain_grad_pointers()
        # for i in range(len(self.grads)):
        #     for g in self.worker_grads:
        #         self.grads[i] += g[i]
        #     self.grads[i] /= stat["num_steps"]

        if self.save_adjacency:
            return stat, np.array(mp_adjacency)
        else:
            return stat
