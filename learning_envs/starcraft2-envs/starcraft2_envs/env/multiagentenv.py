from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MultiAgentEnv(object):

    def init_args(self, parser):
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

    def step(self, actions):
        """Returns reward, terminated, info."""
        raise NotImplementedError

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the size of the observation."""
        raise NotImplementedError

    def get_state(self):
        """Returns the global state."""
        raise NotImplementedError

    def get_state_size(self):
        """Returns the size of the global state."""
        raise NotImplementedError

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        raise NotImplementedError

    def reset(self):
        """Returns initial observations and states."""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "obs_alone_shape": self.get_obs_alone_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
