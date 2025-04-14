import argparse

import numpy as np
import torch

from ctdcomm import envs
from ctdcomm.action_utils import parse_action_args
from ctdcomm.utils import init_args_for_env


def parse_config_args():
    """Parse arguments for the script.

    Returns
    -------
    args
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PyTorch RL trainer")
    # training
    # note: number of steps per epoch = epoch_size X batch_size x nprocesses
    parser.add_argument(
        "--num_epochs", default=100, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=10,
        help="number of update iterations in an epoch",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="number of steps before each update (per thread)",
    )
    parser.add_argument(
        "--nprocesses", type=int, default=1, help="How many processes to run"
    )
    parser.add_argument(
        "--nthreads_per_process", type=int, default=0, help="How many OpenMP threads to use per process, default lets PyTorch device"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Use CUDA acceleration"
    )
    # model
    parser.add_argument("--hid_size", default=64, type=int, help="hidden layer size")
    parser.add_argument(
        "--qk_hid_size",
        default=16,
        type=int,
        help="key and query size for soft attention",
    )
    parser.add_argument(
        "--value_hid_size",
        default=32,
        type=int,
        help="value size for soft attention. Note: current code (at least Dec-/TarMAC) break unless this is the same as hid_size",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        default=False,
        help="make the model recurrent in time",
    )

    # optimization
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor")
    parser.add_argument("--tau", type=float, default=1.0, help="gae (remove?)")
    parser.add_argument(
        "--seed", type=int, default=-1, help="random seed. Pass -1 for random seed"
    )  # TODO: works in thread?
    parser.add_argument(
        "--normalize_rewards",
        action="store_true",
        default=False,
        help="normalize rewards in each batch",
    )
    parser.add_argument("--lrate", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--entr", type=float, default=0, help="entropy regularization coeff"
    )
    parser.add_argument(
        "--value_coeff", type=float, default=0.01, help="coeff for value loss term"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.97, help="RMSprop optimizer alpha"
    )  # Added by JenniBN
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="RMSprop optimizer epsilon"
    )  # Added by JenniBN
    # environment
    parser.add_argument(
        "--env_name", default="dec_predator_prey", help="name of the environment to run",
        choices=("dec_predator_prey", "predator_prey", "traffic_junction", "grf", "smac")
    )
    parser.add_argument(
        "--max_steps",
        default=20,
        type=int,
        help="force to end the game after this many steps",
    )
    parser.add_argument(
        "--nactions",
        default="1",
        type=str,
        help="the number of agent actions (0 for continuous). Use N:M:K for multiple actions",
    )
    parser.add_argument(
        "--action_scale", default=1.0, type=float, help="scale action output from model"
    )
    parser.add_argument(
        "--env_seed",
        type=int,
        default=-1,
        help="random seed for the environment. Pass -1 for random seed",
    )
    # other
    parser.add_argument(
        "--plot", action="store_true", default=False, help="plot training progress"
    )
    parser.add_argument("--plot_env", default="main", type=str, help="plot env name")
    parser.add_argument("--plot_port", default="8097", type=str, help="plot port")
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="save the model after training",
    )
    parser.add_argument(
        "--save_adjacency",
        action="store_true",
        default=False,
        help="save the communication network data whenever saving the model",
    )
    parser.add_argument(
        "--save_every",
        default=0,
        type=int,
        help="save the model after every n_th epoch",
    )
    parser.add_argument("--load", default="", type=str, help="load the model")
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Display environment state",
    )
    parser.add_argument(
        "--random", action="store_true", default=False, help="enable random model"
    )

    # CommNet specific args
    parser.add_argument(
        "--commnet", action="store_true", default=False, help="enable commnet model"
    )
    parser.add_argument(
        "--ic3net", action="store_true", default=False, help="enable ic3net model"
    )
    parser.add_argument(
        "--tarcomm",
        action="store_true",
        default=False,
        help="enable tarmac model (with commnet or ic3net)",
    )
    parser.add_argument(
        "--gacomm", action="store_true", default=False, help="enable gacomm model"
    )
    parser.add_argument(
        "--magic", action="store_true", default=False, help="enable magic model"
    )
    parser.add_argument(
        "--cave", action="store_true", default=False, help="enable the CAVE value head"
    )
    parser.add_argument(
        "--message_augment",
        action="store_true",
        default=False,
        help="enable the critic to be augmented with the aggregated messages received by each agent",
    )
    parser.add_argument(
        "--v_augment",
        action="store_true",
        default=False,
        help="enable the critic to be augmented with the attention value message sent by each agent",
    )
    parser.add_argument(
        "--dec_tarmac",
        action="store_true",
        default=False,
        help="enable dec-tarmac model. Use this with cave and message_augment for CTDComm",
    )
    parser.add_argument(
        "--nagents", type=int, default=1, help="Number of agents (used in multiagent)"
    )
    parser.add_argument(
        "--comm_mode",
        type=str,
        default="avg",
        help="Type of mode for communication tensor calculation [avg|sum]",
    )
    parser.add_argument(
        "--comm_passes",
        type=int,
        default=1,
        help="Number of comm passes per step over the model",
    )
    parser.add_argument(
        "--comm_mask_zero",
        action="store_true",
        default=False,
        help="Whether communication should be there",
    )
    parser.add_argument(
        "--mean_ratio",
        default=1.0,
        type=float,
        help="how much coooperative to do? 1.0 means fully cooperative",
    )
    parser.add_argument(
        "--rnn_type", default="MLP", type=str, help="type of rnn to use. [LSTM|MLP]"
    )
    parser.add_argument(
        "--detach_gap",
        default=10000,
        type=int,
        help="detach hidden state and cell state for rnns at this interval."
        + " Default 10000 (very high)",
    )
    parser.add_argument(
        "--comm_init",
        default="uniform",
        type=str,
        help="how to initialise comm weights [uniform|zeros]",
    )
    parser.add_argument(
        "--hard_attn",
        default=False,
        action="store_true",
        help="Whether to use hard attention: action - talk|silent",
    )
    parser.add_argument(
        "--comm_action_one",
        default=False,
        action="store_true",
        help="Whether to always talk, sanity check for hard attention.",
    )
    parser.add_argument(
        "--advantages_per_action",
        default=False,
        action="store_true",
        help="Whether to multipy log prob for each chosen action with advantages",
    )
    parser.add_argument(
        "--share_weights",
        default=False,
        action="store_true",
        help="Share weights between communication modules between rounds",
    )

    # CommNet specific args
    parser.add_argument(
        "--directed",
        action="store_true",
        default=False,
        help="whether the communication graph is directed",
    )
    parser.add_argument(
        "--self_loop_type1",
        default=2,
        type=int,
        help="self loop type in the first gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)",
    )
    parser.add_argument(
        "--self_loop_type2",
        default=2,
        type=int,
        help="self loop type in the second gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)",
    )
    parser.add_argument(
        "--gat_num_heads",
        default=1,
        type=int,
        help="number of heads in gat layers except the last one",
    )
    parser.add_argument(
        "--gat_num_heads_out",
        default=1,
        type=int,
        help="number of heads in output gat layer",
    )
    parser.add_argument(
        "--gat_hid_size", default=64, type=int, help="hidden size of one head in gat"
    )
    parser.add_argument(
        "--ge_num_heads", default=4, type=int, help="number of heads in the gat encoder"
    )
    parser.add_argument(
        "--first_gat_normalize",
        action="store_true",
        default=False,
        help="whether normalize the coefficients in the first gat layer of the message processor",
    )
    parser.add_argument(
        "--second_gat_normalize",
        action="store_true",
        default=False,
        help="whether normilize the coefficients in the second gat layer of the message proccessor",
    )
    parser.add_argument(
        "--gat_encoder_normalize",
        action="store_true",
        default=False,
        help="whether normilize the coefficients in the gat encoder (they have been normalized if the input graph is complete)",
    )
    parser.add_argument(
        "--use_gat_encoder",
        action="store_true",
        default=False,
        help="whether use the gat encoder before learning the first graph",
    )
    parser.add_argument(
        "--gat_encoder_out_size",
        default=64,
        type=int,
        help="hidden size of output of the gat encoder",
    )
    parser.add_argument(
        "--first_graph_complete",
        action="store_true",
        default=False,
        help="whether the first communication graph is set to a complete graph",
    )
    parser.add_argument(
        "--second_graph_complete",
        action="store_true",
        default=False,
        help="whether the second communication graph is set to a complete graph",
    )
    parser.add_argument(
        "--learn_second_graph",
        action="store_true",
        default=False,
        help="whether learn a new communication graph at the second round of communication",
    )
    parser.add_argument(
        "--message_encoder",
        action="store_true",
        default=False,
        help="whether use the message encoder",
    )
    parser.add_argument(
        "--message_decoder",
        action="store_true",
        default=False,
        help="whether use the message decoder",
    )

    init_args_for_env(parser)
    args = parser.parse_args()

    if args.cuda and args.nprocesses > 1:
        raise RuntimeError("CUDA is not compatible with multiprocessing (using --nprocesses > 1)")

    if args.cave:
        args.save_adjacency = True

    if args.commnet and not (
        args.dec_tarmac or args.tarcomm or args.ic3net or args.gacomm
    ):
        args.save_adjacency = 0

    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0

        # For TJ set comm action to 1 as specified in paper to showcase
        # importance of individual rewards even in cooperative games
        if args.env_name == "traffic_junction":
            args.comm_action_one = True

    if args.gacomm:
        args.commnet = 1
        args.mean_ratio = 0
        if args.env_name == "traffic_junction":
            args.comm_action_one = True

    if args.magic:
        args.recurrent = 1

    # Enemy comm
    args.nfriendly = args.nagents
    if (hasattr(args, "enemy_comm") and args.enemy_comm) or (
        hasattr(args, "learning_prey") and args.learning_prey
    ):
        if hasattr(args, "nenemies"):
            args.nagents += args.nenemies
        else:
            raise RuntimeError("Env. needs to pass argument 'nenemies'.")

    # TODO: need to understand what is happening here
    if args.env_name == "grf":
        render = args.render
        args.render = False
    else:
        render = None

    env = envs.init(args.env_name, args, False)

    # TODO: Check that observation dim works with the new api
    num_inputs = env.observation_dim
    if args.env_name == "dec_predator_prey":
        args.num_actions = env.naction  # [env.naction]
        args.dim_actions = 1
    else:
        args.num_actions = env.num_actions
        args.dim_actions = env.dim_actions

    # Multi-action
    if not isinstance(args.num_actions, (list, tuple)):  # single action case
        args.num_actions = [args.num_actions]
    args.num_inputs = num_inputs

    # Hard attention
    if args.hard_attn and args.commnet:
        # add comm_action as last dim in actions
        args.num_actions = list(args.num_actions) + [2]
        args.dim_actions = args.dim_actions + 1

    # Recurrence
    if (args.commnet or args.magic) and (args.recurrent or args.rnn_type == "LSTM"):
        args.recurrent = True
        args.rnn_type = "LSTM"

    parse_action_args(args)

    if args.seed == -1:
        args.seed = np.random.randint(0, 10000)
    torch.manual_seed(args.seed)
    if args.env_seed == -1:
        if args.env_name == "dec_predator_prey":
            args.env_seed = None  # the environment has a nie way of dealing with seeds
        else:
            args.env_seed = np.random.randint(0, 10000)

    # print(args)

    return args, env, render
