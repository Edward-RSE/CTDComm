import gym
import starcraft2_envs

from ctdcomm.env_wrappers import GymWrapper

def init(env_name, args, final_init=True):
    if env_name == 'dec_predator_prey':
        import predator_prey
        env = predator_prey.env.PredatorPreyEnv(args)
        if args.display:
            env.init_curses()
    elif env_name == "predator_prey":
        env = gym.make("PredatorPrey-v0", disable_env_checker=True)  # JenniBN, edited to work with latest gym
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == "traffic_junction":
        env = gym.make("TrafficJunction-v0", disable_env_checker=True)  # JenniBN, edited to work with latest gym
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == "grf":
        env = gym.make('GRFWrapper-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == "starcraft2":
        env = starcraft2_envs.StarCraft2Env(args)
    else:
        raise RuntimeError("wrong env name")

    return env
