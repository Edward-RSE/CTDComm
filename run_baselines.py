import os
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import visdom

from ctdcomm import data
from ctdcomm.multi_processing import MultiProcessTrainer
from ctdcomm.policy_nets.comm import CommNetMLP
from ctdcomm.policy_nets.dec_tarmac import DecTarMAC
from ctdcomm.policy_nets.ga_comm import GACommNetMLP
from ctdcomm.policy_nets.magic import MAGIC
from ctdcomm.policy_nets.models import MLP, RNN, Random
from ctdcomm.policy_nets.tar_comm import TarCommNetMLP
from ctdcomm.trainer import Trainer
from ctdcomm.utils import LogField, display_models, merge_stat
from ctdcomm.config import parse_config_args

warnings.simplefilter("error")


def init_torch():
    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True
    torch.set_default_dtype(torch.double)
    torch.multiprocessing.set_start_method("spawn", force=True)


def load_model(path, policy_net, trainer, log):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d["policy_net"])
    log.update(d["log"])
    trainer.load_state_dict(d["trainer"])


def save_model(policy_net, trainer, log, run_dir, final, episode=0):
    d = dict()
    d["policy_net"] = policy_net.state_dict()
    d["log"] = log
    d["trainer"] = trainer.state_dict()
    if final:
        model_filename = run_dir / "model.pt"

        i = 0
        while os.path.exists(model_filename):
            i += 1
            model_filename = run_dir / ("model%i.pt" % (i))
        torch.save(d, model_filename)
    else:
        model_filename = run_dir / ("model_ep%i.pt" % (episode))

        i = 0
        while os.path.exists(model_filename):
            i += 1
            model_filename = run_dir / ("model_ep%i_%i.pt" % (i, episode))
        torch.save(d, model_filename)


def signal_handler(env, env_name, display):
    def handler(signal, frame):
        print("You pressed Ctrl+C! Exiting gracefully.")
        if "dec" in env_name:
            env.close()
        else:
            if display:
                env.exit_render()
        sys.exit(0)

    return handler


def get_env_name(args):
    if args.env_name == "traffic_junction":
        env_name_str = args.env_name + "_" + args.difficulty
        if args.difficulty == "hard" and args.add_rate_min == args.add_rate_max:
            if args.add_rate_max == 0.1:
                env_name_str = env_name_str + "_add_01"
            elif args.add_rate_max == 0.2:
                env_name_str = env_name_str + "_add_02"
    elif "predator_prey" in args.env_name:
        if "dec" in args.env_name:
            env_name_str = args.env_name + f"_{args.nagents}v{args.nenemies}"

            if args.comm_range != 0:
                env_name_str = args.env_name + "_cr=" + str(args.comm_range)

            if args.learning_prey:
                env_name_str = env_name_str + "_learning_prey"
            elif args.moving_prey:
                env_name_str = env_name_str + "_random_prey"
        else:
            env_name_str = args.env_name
            if args.nagents == 5:
                env_name_str = env_name_str + "_5v1"  #'_medium'
            elif args.nagents == 10:
                if args.nenemies == 1:
                    env_name_str = env_name_str + "_10v1"  #'_hard'
                if args.nenemies == 2:
                    env_name_str = env_name_str + "_10v2"
            elif args.nagents == 20:
                env_name_str = env_name_str + "_20v1"
    else:
        env_name_str = args.env_name

    return env_name_str


def get_run_dir(args, env_name_str):
    model_dir = Path("./ctdcomm_saved") / env_name_str
    if args.magic:
        model_dir = model_dir / "magic"
    elif args.gacomm:
        model_dir = model_dir / "gacomm"
    elif args.dec_tarmac:
        if args.ic3net:
            if args.cave:
                if args.message_augment:
                    model_dir = model_dir / "ctdcomm"
                elif args.v_augment:
                    model_dir = model_dir / "ctdcomm_v_aug"
            else:
                if args.message_augment:
                    model_dir = model_dir / "dec_tarmac_message_aug"
                elif args.v_augment:
                    model_dir = model_dir / "dec_tarmac_v_aug"
                else:
                    model_dir = model_dir / "dec_tarmac"
        elif args.commnet:
            model_dir = model_dir / "dec_tarcomm"
    elif args.tarcomm:
        if args.ic3net:
            model_dir = model_dir / "tar_ic3net"
        elif args.commnet:
            model_dir = model_dir / "tar_commnet"
        else:
            model_dir = model_dir / "other"

        if args.comm_passes != 1:
            model_dir = Path(
                str(model_dir) + "_" + str(args.comm_passes) + "comm_rounds"
            )
    elif args.ic3net:
        model_dir = model_dir / "ic3net"
    elif args.commnet:
        model_dir = model_dir / "commnet"
    else:
        model_dir = model_dir / "other"

    if args.cave and not (args.message_augment or args.v_augment):
        # Alter the dir name to differentiate from a standard value head
        dir_head, dir_tail = os.path.split(model_dir)
        model_dir = Path(dir_head + "/" + dir_tail + "_cave")

    if args.env_name == "grf":
        model_dir = model_dir / args.scenario

    if args.load:
        run_dir = args.load[: args.load.rfind("/")]
        curr_run = run_dir[run_dir.rfind("/") + 1 :]
        run_dir = Path(run_dir)
    else:
        curr_run = "run%i" % args.seed
        if (model_dir / curr_run).exists():
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in model_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = model_dir / curr_run

    return run_dir


def get_policy_net(args):
    """Return the requested Policy Net from the command line arguments."""
    num_inputs = args.num_inputs
    if args.magic:
        policy_net = MAGIC(args, num_inputs)
    elif args.gacomm:
        policy_net = GACommNetMLP(args, num_inputs)
    elif args.commnet:
        if args.tarcomm:
            policy_net = TarCommNetMLP(args, num_inputs)
        elif args.dec_tarmac:
            policy_net = DecTarMAC(args, num_inputs)
        else:
            policy_net = CommNetMLP(args, num_inputs)
    elif args.random:
        policy_net = Random(args, num_inputs)
    elif args.recurrent:
        policy_net = RNN(args, num_inputs)
    else:
        policy_net = MLP(args, num_inputs)

    return policy_net


def run(args, policy_net, trainer, log, run_dir, vis, num_epochs):
    num_episodes = 0
    if args.save and not args.load:
        os.makedirs(run_dir)

        # Save the config as a separate file
        with (run_dir / "config.json").open(mode="w") as f:
            for arg in vars(args):
                f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

    np.set_printoptions(precision=2)

    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            if args.save_adjacency:
                s, adjacency_data = trainer.train_batch(ep)
            else:
                s = trainer.train_batch(ep)
            print("batch: ", n)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log["epoch"].data) + 1
        num_episodes += stat["num_episodes"]
        for k, v in log.items():
            if k == "epoch":
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        print("Epoch {}".format(epoch))
        print("Episode: {}".format(num_episodes))
        print("Reward: {}".format(stat["reward"]))
        print("Time: {:.2f}s".format(epoch_time))

        if "enemy_reward" in stat.keys():
            print("Enemy-Reward: {}".format(stat["enemy_reward"]))
        if "add_rate" in stat.keys():
            print("Add-Rate: {:.2f}".format(stat["add_rate"]))
        if "success" in stat.keys():
            print("Success: {:.4f}".format(stat["success"]))
        if "steps_taken" in stat.keys():
            print("Steps-Taken: {:.2f}".format(stat["steps_taken"]))
        if "comm_action" in stat.keys():
            print("Comm-Action: {}".format(stat["comm_action"]))
        if "enemy_comm" in stat.keys():
            print("Enemy-Comm: {}".format(stat["enemy_comm"]))
        if "density1" in stat.keys():
            print("density1: {:.4f}".format(stat["density1"]))
        if "density2" in stat.keys():
            print("density2: {:.4f}".format(stat["density2"]))

        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(
                        np.asarray(v.data),
                        np.asarray(log[v.x_axis].data[-len(v.data) :]),
                        win=k,
                        opts=dict(xlabel=v.x_axis, ylabel=k),
                    )

        if args.save_every and ep and args.save and ep % args.save_every == 0:
            save_model(policy_net, trainer, log, run_dir, final=False, episode=ep)
            if args.save_adjacency:
                adj_filename = run_dir / ("adjacency_epoch_%i.npy" % (ep))
                i = 0
                while os.path.exists(adj_filename):
                    i += 1
                    adj_filename = run_dir / ("adjacency_epoch_%i_%d.npy" % (ep, i))
                print("Saving adjacency data to", adj_filename)
                print("\t", np.array(adjacency_data).shape)
                np.save(adj_filename, adjacency_data)

    if args.save:  # JenniBN - moved this an indent lower so it isn't saving every epoch
        save_model(policy_net, trainer, log, run_dir, final=True)
        if args.save_adjacency:
            adj_filename = run_dir / "adjacency_final_epoch.npy"
            i = 0
            while os.path.exists(adj_filename):
                i += 1
                adj_filename = run_dir / ("adjacency_final_epoch%i.npy" % (i))
            print("Doing the final adjacency data save to", adj_filename)
            print("\t", np.array(adjacency_data).shape)
            np.save(adj_filename, adjacency_data)


def run_baselines():
    """Main entry point for `run_baselines.py`."""
    init_torch()
    args, env, render = parse_config_args()
    print(args)
    signal.signal(signal.SIGINT, signal_handler(env, args.env_name, args.display))
    policy_net = get_policy_net(args)

    if args.env_name == "grf":
        args.render = render

    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA has been requested, but is not available")
    else:
        device = torch.device("cpu")

    if args.nprocesses > 1:
        trainer = MultiProcessTrainer(
            args, lambda: Trainer(args, policy_net, data.init(args.env_name, args))
        )
    else:
        trainer = Trainer(args, policy_net, data.init(args.env_name, args), device=device)

    log = dict()
    log["epoch"] = LogField(list(), False, None, None)
    log["reward"] = LogField(list(), True, "epoch", "num_episodes")
    log["enemy_reward"] = LogField(list(), True, "epoch", "num_episodes")
    log["success"] = LogField(list(), True, "epoch", "num_episodes")
    log["steps_taken"] = LogField(list(), True, "epoch", "num_episodes")
    log["add_rate"] = LogField(list(), True, "epoch", "num_episodes")
    log["comm_action"] = LogField(list(), True, "epoch", "num_steps")
    log["enemy_comm"] = LogField(list(), True, "epoch", "num_steps")
    log["value_loss"] = LogField(list(), True, "epoch", "num_steps")
    log["action_loss"] = LogField(list(), True, "epoch", "num_steps")
    log["entropy"] = LogField(list(), True, "epoch", "num_steps")
    log["density1"] = LogField(list(), True, "epoch", "num_steps")
    log["density2"] = LogField(list(), True, "epoch", "num_steps")

    if args.load != "":
        load_model(args.load, policy_net, trainer, log)

    if not args.display:
        display_models([policy_net])

    if args.plot:
        vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)
    else:
        vis = None

    env_name_str = get_env_name(args)
    run_dir = get_run_dir(args, env_name_str)

    run(args, policy_net, trainer, log, run_dir, vis, args.num_epochs)

    if args.display:
        # The MAGIC code called a fucntion called 'env.end_display()' which didn't exist in any of the environments...
        if "dec" in args.env_name:
            env.close()
        else:
            if args.display:
                env.exit_render()

    if args.save:
        save_model(policy_net, trainer, log, run_dir, final=True)

    if sys.flags.interactive == 0 and args.nprocesses > 1:
        trainer.quit()
        os._exit(0)
    else:
        trainer.env.close()


if __name__ == "__main__":
    run_baselines()
