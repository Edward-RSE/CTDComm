import numbers
import math
from collections import namedtuple

import predator_prey
import numpy as np

import gym
import ic3net_envs
import torch
import time
import sys


LogField = namedtuple('LogField', ('data', 'plot', 'x_axis', 'divide_by'))


def init_torch():
    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True
    torch.set_default_dtype(torch.double)

    
def merge_stat(src, dest):
    for k, v in src.items():
        if not k in dest:
            dest[k] = v
        elif isinstance(v, numbers.Number):
            dest[k] = dest.get(k, 0) + v
        elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
            dest[k] = dest.get(k, 0) + v  # for rewards in case of multi-agent
        else:
            if isinstance(dest[k], list) and isinstance(v, list):
                dest[k].extend(v)
            elif isinstance(dest[k], list):
                dest[k].append(v)
            else:
                dest[k] = [dest[k], v]


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def multinomials_log_density(actions, log_probs):
    log_prob = 0
    for i in range(len(log_probs)):
        log_prob += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    return log_prob


def multinomials_log_densities(actions, log_probs):
    log_prob = [0] * len(log_probs)
    for i in range(len(log_probs)):
        log_prob[i] += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    log_prob = torch.cat(log_prob, dim=-1)
    return log_prob


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


class Timer:
    def __init__(self, msg, sync=False):
        self.msg = msg
        self.sync = sync

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("{}: {} s".format(self.msg, self.interval))


def pca(X, k=2):
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])


def init_args_for_smac(parser):
    """Initialise a parser for the SMAC environment.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.

    """
    env_args = parser.add_argument_group('SMAC')
    env_args.add_argument("--smac_capability_config", type=str, required=True,
                          help="Path to a YAML file containing a SMAC capability configuration.")

    env_args.add_argument("--smac_map_name", default="8m",
                          help="Name of the SMAC map to use.")
    env_args.add_argument("--smac_step_mul", default=8, type=int,
                          help="Game steps per agent step.")
    env_args.add_argument("--smac_move_amount", default=2, type=int,
                          help="Distance units move per step.")
    env_args.add_argument("--smac_difficulty", default="7", type=str,
                          help="Difficulty level of built-in AI.")
    env_args.add_argument("--smac_game_version", default=None, type=str,
                          help="SC2 version to use. None = latest.")
    env_args.add_argument("--smac_continuing_episode", default=False, action='store_true',
                          help="Episode continues after time limit.")
    env_args.add_argument("--smac_obs_all_health", default=True, action='store_true',
                          help="Observations include all units' health.")
    env_args.add_argument("--smac_obs_own_health", default=True, action='store_true',
                          help="Observations include only own health (ignored if all health enabled).")
    env_args.add_argument("--smac_obs_last_action", default=False, action='store_true',
                          help="Include last actions of units in observations.")
    env_args.add_argument("--smac_obs_pathing_grid", default=False, action='store_true',
                          help="Include pathing grid in observations.")
    env_args.add_argument("--smac_obs_terrain_height", default=False, action='store_true',
                          help="Include terrain height in observations.")
    env_args.add_argument("--smac_obs_instead_of_state", default=False, action='store_true',
                          help="Use combined observations as global state.")
    env_args.add_argument("--smac_obs_timestep_number", default=False, action='store_true',
                          help="Include timestep in observations.")
    env_args.add_argument("--smac_obs_own_pos", default=False, action='store_true',
                          help="Include own position in observations.")
    env_args.add_argument("--smac_obs_starcraft", default=True, action='store_true',
                          help="Enable standard StarCraft observations.")
    env_args.add_argument("--smac_conic_fov", default=False, action='store_true',
                          help="Enable conic field of view for agents.")
    env_args.add_argument("--smac_num_fov_actions", default=12, type=int,
                          help="Number of actions in discretised field of view.")
    env_args.add_argument("--smac_state_last_action", default=True, action='store_true',
                          help="Include last actions in global state.")
    env_args.add_argument("--smac_state_timestep_number", default=False, action='store_true',
                          help="Include timestep in global state.")
    env_args.add_argument("--smac_reward_sparse", default=False, action='store_true',
                          help="Use sparse rewards (1/-1 for win/loss).")
    env_args.add_argument("--smac_reward_only_positive", default=True, action='store_true',
                          help="Restrict all rewards to be positive.")
    env_args.add_argument("--smac_reward_death_value", default=10, type=int,
                          help="Reward for killing enemy or penalty for death.")
    env_args.add_argument("--smac_reward_win", default=200, type=int,
                          help="Reward for winning an episode.")
    env_args.add_argument("--smac_reward_defeat", default=0, type=int,
                          help="Reward for losing an episode.")
    env_args.add_argument("--smac_reward_negative_scale", default=0.5, type=float,
                          help="Scale factor for negative rewards.")
    env_args.add_argument("--smac_reward_scale", default=True, action='store_true',
                          help="Enable reward scaling.")
    env_args.add_argument("--smac_reward_scale_rate", default=20, type=int,
                          help="Rate used when scaling rewards.")
    env_args.add_argument("--smac_use_unit_ranges", default=False, action='store_true',
                          help="Use per-unit attack range info.")
    env_args.add_argument("--smac_min_attack_range", default=2, type=int,
                          help="Minimum attack range for ranged units.")
    env_args.add_argument("--smac_kill_unit_step_mul", default=2, type=int,
                          help="Steps before reward for unit kill is assigned.")
    env_args.add_argument("--smac_fully_observable", default=False, action='store_true',
                          help="Environment is fully observable.")
    env_args.add_argument("--smac_replay_dir", default="", type=str,
                          help="Directory to save replays. Empty = SC2 default.")
    env_args.add_argument("--smac_replay_prefix", default="", type=str,
                          help="Prefix for saved replay filenames.")
    env_args.add_argument("--smac_heuristic_ai", default=False, action='store_true',
                          help="Enable a non-learning heuristic AI.")
    env_args.add_argument("--smac_heuristic_rest", default=False, action='store_true',
                          help="Restrict heuristic AI actions to those available to RL agents.")
    env_args.add_argument("--smac_prob_obs_enemy", default=1.0, type=float,
                          help="Probability of observing an enemy in range.")
    env_args.add_argument("--smac_action_mask", default=True, action='store_true',
                          help="Mask unavailable actions.")


def init_args_for_env(parser):
    env_dict = {
        # EP: I believe these two are not used anymore, as I can't find any references to them
        # 'levers': 'Levers-v0',
        # 'number_pairs': 'NumberPairs-v0',
        'predator_prey': 'PredatorPrey-v0',
        'dec_predator_prey': 'PredatorPrey-v1',
        'traffic_junction': 'TrafficJunction-v0',
        'grf': 'GRFWrapper-v0',
        'smac': 'SMAC-v2'
    }

    args = sys.argv
    env_name = None
    for index, item in enumerate(args):
        if item == '--env_name':
            env_name = args[index + 1]

    if not env_name or env_name not in env_dict:
        return

    # Unfortunately, we need to make a nested IF here to deal with the fact that
    # the SMAC environment does not have an `init_args` method.
    if env_dict[env_name] == "SMAC-v2":
        init_args_for_smac(parser)
    else:
        if env_dict[env_name] == 'PredatorPrey-v1':
            env = predator_prey.env.PredatorPreyEnv()
        else:
            env = gym.make(env_dict[env_name], disable_env_checker=True) #JenniBN, edited to work with latest gym

        env.init_args(parser)


def display_models(list_models):
    print('='*100)
    print('Model log:\n')
    for model in list_models:
        print(model)
    print('='*100 + '\n')
