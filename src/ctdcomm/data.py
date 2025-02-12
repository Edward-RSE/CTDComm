import sys
# import gym #I'm importing gym in the body to avoid importing unnecessarily
from ctdcomm.env_wrappers import *

def init(env_name, args, final_init=True):
    if env_name == 'dec_predator_prey':
        import predator_prey
        env = predator_prey.env.PredatorPreyEnv(args)
        if args.display:
            env.init_curses()
    elif env_name == 'predator_prey':
        import gym
        import ic3net_envs
        env = gym.make('PredatorPrey-v0', disable_env_checker=True) #JenniBN, edited to work with latest gym
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        import gym
        import ic3net_envs
        env = gym.make('TrafficJunction-v0', disable_env_checker=True) #JenniBN, edited to work with latest gym
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'grf':
        import gym
        import ic3net_envs
        env = gym.make('GRFWrapper-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env)

    else:
        raise RuntimeError("wrong env name")

    return env
