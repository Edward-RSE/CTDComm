import time
from utils import *
import torch
import torch.multiprocessing as mp

class MultiProcessWorker(mp.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, rank, trainer_maker, comm, seed, save_adjacency=False, *args, **kwargs):
        self.rank = rank
        self.seed = seed
        self.save_adjacency = save_adjacency
        super(MultiProcessWorker, self).__init__()
        self.trainer = trainer_maker()
        self.comm = comm

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            self.device = f"cuda:{self.rank % num_devices}"
        else:
            self.device = "cpu"


    def run(self):
        torch.manual_seed(self.seed + self.rank + 1)
        np.random.seed(self.seed + self.rank + 1)

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, epoch = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                if self.save_adjacency:
                    batch, stat, batch_adjacency = self.trainer.run_batch(epoch)
                else:
                    batch, stat = self.trainer.run_batch(epoch)
                self.trainer.optimizer.zero_grad()
                s = self.trainer.compute_grad(batch)
                merge_stat(s, stat)
                if self.save_adjacency:
                    self.comm.send([stat, batch_adjacency])
                else:
                    self.comm.send(stat)
            elif task == 'send_grads':
                grads = []
                for p in self.trainer.params:
                    if p._grad is not None:
                        grads.append(p._grad.data)

                self.comm.send(grads)


class MultiProcessTrainer(object):
    def __init__(self, args, trainer_maker):
        self.comms = []
        self.trainer = trainer_maker()
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1
        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, trainer_maker, comm_remote, args.seed, args.save_adjacency)
            worker.start()
        self.grads = None
        self.worker_grads = None
        self.is_random = args.random
        self.save_adjacency = args.save_adjacency

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def obtain_grad_pointers(self):
        # only need perform this once
        if self.grads is None:
            self.grads = []
            for p in self.trainer.params:
                if p._grad is not None:
                    self.grads.append(p._grad.data)

        if self.worker_grads is None:
            self.worker_grads = []
            for comm in self.comms:
                comm.send('send_grads')
                self.worker_grads.append(comm.recv())

    def train_batch(self, epoch):
        # run workers in parallel
        for comm in self.comms:
            comm.send(['run_batch', epoch])

        # run its own trainer
        if self.save_adjacency:
            batch, stat, batch_adjacency = self.trainer.run_batch(epoch)
            mp_adjacency = list(batch_adjacency)
        else:
            batch, stat = self.trainer.run_batch(epoch)
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(batch)
        merge_stat(s, stat)

        # check if workers are finished
        for comm in self.comms:
            if self.save_adjacency:
                s, batch_adjacency = comm.recv()
                mp_adjacency += list(batch_adjacency)
            else:
                s = comm.recv()
            merge_stat(s, stat)

        # add gradients of workers
        self.obtain_grad_pointers()
        for i in range(len(self.grads)):
            for g in self.worker_grads:
                self.grads[i] += g[i]
            self.grads[i] /= stat['num_steps']

        self.trainer.optimizer.step()
        if self.save_adjacency:
            return stat, np.array(mp_adjacency)
        else:
            return stat

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)
