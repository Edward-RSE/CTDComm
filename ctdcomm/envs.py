import gym
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
        distribution_config = {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["medivac"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        }
        env = StarCraftCapabilityEnvWrapper(
            capability_config=distribution_config,
            map_name=args.smac_challenge,
            debug=True,
            seed=args.seed,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
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
