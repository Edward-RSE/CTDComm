import gym
import yaml
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from ctdcomm.env_wrappers import GymWrapper


def init(env_name, args, final_init=True):
    if env_name == "dec_predator_prey":
        import predator_prey

        env = predator_prey.env.PredatorPreyEnv(args)
        if args.display:
            env.init_curses()
    elif env_name == "predator_prey":
        env = gym.make(
            "PredatorPrey-v0", disable_env_checker=True
        )  # JenniBN, edited to work with latest gym
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == "traffic_junction":
        env = gym.make(
            "TrafficJunction-v0", disable_env_checker=True
        )  # JenniBN, edited to work with latest gym
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == "grf":
        env = gym.make("GRFWrapper-v0")
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == "smac":
        with open(args.smac_capability_config) as file_in:
             args.smac_capability_config = yaml.safe_load(file_in)
        smac_args = {k[5:]: v for k, v in vars(args).items() if k.startswith("smac_")}
        env = StarCraftCapabilityEnvWrapper(
            seed=args.seed,
            **smac_args,
        )
        env_info = env.get_env_info()
        env.observation_dim = env_info["obs_shape"]
        env.num_actions = env_info["n_actions"]
        env.dim_actions = 1
        if env_info["n_agents"] != args.nagents:
            raise ValueError(
                f"Invalid number of agents, {args.nagents} requested but {env_info['n_agents']} in StarCraft2Env"
            )
    else:
        raise RuntimeError("wrong env name")

    return env
