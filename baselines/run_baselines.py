import sys
import time
import signal
import argparse
import os
from pathlib import Path
import json

import numpy as np
import torch
import visdom
from models import *
from comm import CommNetMLP
from tar_comm import TarCommNetMLP
from ga_comm import GACommNetMLP
from magic import MAGIC
from dec_tarmac import DecTarMAC
from trainer import Trainer

sys.path.append("..") 
import data
from utils import *
from action_utils import parse_action_args
from multi_processing import MultiProcessTrainer
# import gym

# gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--qk_hid_size', default=16, type=int,
                    help='key and query size for soft attention')
parser.add_argument('--value_hid_size', default=32, type=int,
                    help='value size for soft attention')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
parser.add_argument('--alpha', type=float, default=0.97,
                    help='RMSprop optimizer alpha') #Added by JenniBN
parser.add_argument('--eps', type=float, default=1e-6,
                    help='RMSprop optimizer epsilon') #Added by JenniBN
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--plot_port', default='8097', type=str,
                    help='plot port')
parser.add_argument('--save', action="store_true", default=False,
                    help='save the model after training')
parser.add_argument('--save_adjacency', action="store_true", default=False,
                    help='save the communication network data whenever saving the model')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable ic3net model")
parser.add_argument('--tarcomm', action='store_true', default=False,
                    help="enable tarmac model (with commnet or ic3net)")
parser.add_argument('--gacomm', action='store_true', default=False,
                    help="enable gacomm model")
parser.add_argument('--magic', action='store_true', default=False,
                    help="enable magic model")
parser.add_argument('--cave', action='store_true', default=False,
                    help="enable the CAVE value head")
parser.add_argument('--dec_tarmac', action='store_true', default=False,
                    help="enable dec-tarmac model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log prob for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')

# CommNet specific args
parser.add_argument('--directed', action='store_true', default=False,
                    help='whether the communication graph is directed')
parser.add_argument('--self_loop_type1', default=2, type=int,
                    help='self loop type in the first gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--self_loop_type2', default=2, type=int,
                    help='self loop type in the second gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--gat_num_heads', default=1, type=int,
                    help='number of heads in gat layers except the last one')
parser.add_argument('--gat_num_heads_out', default=1, type=int,
                    help='number of heads in output gat layer')
parser.add_argument('--gat_hid_size', default=64, type=int,
                    help='hidden size of one head in gat')
parser.add_argument('--ge_num_heads', default=4, type=int,
                    help='number of heads in the gat encoder')
parser.add_argument('--first_gat_normalize', action='store_true', default=False,
                    help='whether normalize the coefficients in the first gat layer of the message processor')
parser.add_argument('--second_gat_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the second gat layer of the message proccessor')
parser.add_argument('--gat_encoder_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the gat encoder (they have been normalized if the input graph is complete)')
parser.add_argument('--use_gat_encoder', action='store_true', default=False,
                    help='whether use the gat encoder before learning the first graph')
parser.add_argument('--gat_encoder_out_size', default=64, type=int,
                    help='hidden size of output of the gat encoder')
parser.add_argument('--first_graph_complete', action='store_true', default=False,
                    help='whether the first communication graph is set to a complete graph')
parser.add_argument('--second_graph_complete', action='store_true', default=False,
                    help='whether the second communication graph is set to a complete graph')
parser.add_argument('--learn_second_graph', action='store_true', default=False,
                    help='whether learn a new communication graph at the second round of communication')
parser.add_argument('--message_encoder', action='store_true', default=False,
                    help='whether use the message encoder')
parser.add_argument('--message_decoder', action='store_true', default=False,
                    help='whether use the message decoder')


init_args_for_env(parser)
args = parser.parse_args()

if args.cave:
    args.save_adjacency = True

if args.commnet and not (args.dec_tarmac or args.tarcomm or args.ic3net or args.gacomm):
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
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemies'.")

if args.env_name == 'grf':
    render = args.render
    args.render = False
env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if (args.commnet or args.magic) and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)

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
        
if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.env_name == 'grf':
    args.render = render
if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()    
    
log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['density1'] = LogField(list(), True, 'epoch', 'num_steps')
log['density2'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)

if args.env_name == 'traffic_junction':
    env_name_str = args.env_name + '_' + args.difficulty
    if args.difficulty == 'hard' and args.add_rate_min == args.add_rate_max:
        if args.add_rate_max == 0.1:
            env_name_str = env_name_str + '_add_01'
        elif args.add_rate_max == 0.2:
            env_name_str = env_name_str + '_add_02'
elif args.env_name == 'predator_prey':
    if args.nagents == 5:
        env_name_str = args.env_name + '_medium'
    elif args.nagents == 10:
        if args.nenemies == 1:
            env_name_str = args.env_name + '_hard'
        if args.nenemies == 2:
            env_name_str = args.env_name + '_10v2'
    elif args.nagents == 20:
        env_name_str = args.env_name + '_20v1'
else:
    env_name_str = args.env_name

model_dir = Path('./saved') / env_name_str
if args.magic:
    model_dir = model_dir / 'magic'
elif args.gacomm:
    model_dir = model_dir / 'gacomm'
elif args.dec_tarmac:
    if args.ic3net:
        model_dir = model_dir / 'dec_tarmac'
    elif args.commnet:
        model_dir = model_dir / 'dec_tarcomm'
elif args.tarcomm:
    if args.ic3net:
        model_dir = model_dir / 'tar_ic3net'
    elif args.commnet:
        model_dir = model_dir / 'tar_commnet'
    else:
        model_dir = model_dir / 'other'
    
    if args.comm_passes != 1:
        model_dir = Path(str(model_dir) + "_" + str(args.comm_passes) + "comm_rounds")
elif args.ic3net:
    model_dir = model_dir / 'ic3net'
elif args.commnet:
    model_dir = model_dir / 'commnet'
else:
    model_dir = model_dir / 'other'

if args.cave:
    # Alter the dir name to differentiate from a standard value head
    dir_head, dir_tail = os.path.split(model_dir)
    model_dir = Path(dir_head + "/" + dir_tail + "_cave")

if args.env_name == 'grf':
    model_dir = model_dir / args.scenario

if args.load:
    run_dir = args.load[:args.load.rfind('/')]
    curr_run = run_dir[run_dir.rfind('/')+1:]
    run_dir = Path(run_dir)
else:
    curr_run = 'run%i' % args.seed
    if (model_dir / curr_run).exists():
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                            model_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run 

def run(num_epochs): 
    num_episodes = 0
    if args.save and not args.load:
        os.makedirs(run_dir)

        # Save the config as a separate file
        with (run_dir / 'config.json').open(mode='w') as f:
            for arg in vars(args):
                f.write(str(arg) + ': ' + str(getattr(args, arg)) + '\n')

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
            print('batch: ', n)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        num_episodes += stat['num_episodes']
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)
        
        print('Epoch {}'.format(epoch))
        print('Episode: {}'.format(num_episodes))
        print('Reward: {}'.format(stat['reward']))
        print('Time: {:.2f}s'.format(epoch_time))
        
        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.4f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-Taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'density1' in stat.keys():
            print('density1: {:.4f}'.format(stat['density1']))
        if 'density2' in stat.keys():
            print('density2: {:.4f}'.format(stat['density2']))        


        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))
    
        if args.save_every and ep and args.save and ep % args.save_every == 0:
            save(final=False, episode=ep)
            if args.save_adjacency:
                adj_filename = run_dir / ("adjacency_epoch_%i.npy" %(ep))
                i=0
                while os.path.exists(adj_filename):
                    i+=1
                    adj_filename = run_dir / ("adjacency_epoch_%i_%d.npy" %(ep, i))
                print("Saving adjacency data to", adj_filename)
                print("\t", np.array(adjacency_data).shape)
                np.save(adj_filename, adjacency_data)

    if args.save: #JenniBN - moved this an indent lower so it isn't saving every epoch
        save(final=True)
        if args.save_adjacency:
            adj_filename = run_dir / "adjacency_final_epoch.npy"
            i=0
            while os.path.exists(adj_filename):
                i+=1
                adj_filename = run_dir / ("adjacency_final_epoch%i.npy" %(i))
            print("Doing the final adjacency data save to", adj_filename)
            print("\t", np.array(adjacency_data).shape)
            np.save(adj_filename, adjacency_data)

def save(final, episode=0): 
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        model_filename = run_dir / 'model.pt'

        i = 0
        while os.path.exists(model_filename):
            i+=1
            model_filename = run_dir / ("model%i.pt" %(i))
        torch.save(d, model_filename)
    else:
        model_filename = run_dir / ('model_ep%i.pt' %(episode))

        i = 0
        while os.path.exists(model_filename):
            i+=1
            model_filename = run_dir / ("model_ep%i_%i.pt" %(i, episode))
        torch.save(d, model_filename)

def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

run(args.num_epochs)
if args.display:
    env.end_display()

if args.save:
    save(final=True)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
