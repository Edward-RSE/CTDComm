def init_args_for_envs(parser):
    """Additional (command line) arguments specific to this environment."""
    env = parser.add_argument_group("StarCraft II")
    env.add_argument(
        "--map_name", type=str, default="3m", help="Which smac map to run on"
    )
    env.add_argument(
        "--eval_map_name", type=str, default="3m", help="Which smac map to eval on"
    )
    env.add_argument(
        "--run_dir", type=str, default="", help="Which smac map to eval on"
    )
    env.add_argument("--add_move_state", action="store_true", default=False)
    env.add_argument("--add_local_obs", action="store_true", default=False)
    env.add_argument("--add_distance_state", action="store_true", default=False)
    env.add_argument("--add_enemy_action_state", action="store_true", default=False)
    env.add_argument("--add_agent_id", action="store_true", default=False)
    env.add_argument("--add_visible_state", action="store_true", default=False)
    env.add_argument("--add_xy_state", action="store_true", default=False)
    env.add_argument("--use_state_agent", action="store_false", default=True)
    env.add_argument("--use_mustalive", action="store_false", default=True)
    env.add_argument("--add_center_xy", action="store_false", default=True)
    env.add_argument("--random_agent_order", action="store_true", default=False)
    env.add_argument("--sight_range", type=int, default=9)
    env.add_argument("--shoot_range", type=int, default=6)
    env.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )
    env.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    env.add_argument(
        "--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )

    return parser
