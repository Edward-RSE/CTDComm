#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a predator prey environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision sqaure around it.
Adapted from the IC3Net env for use in PettingZoo

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for prey agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, preys & units >
"""

# core modules
import numpy as np
import functools
import curses

# seed utils
import predator_prey.env.seeding as seeding

# Gymnasium and PettingZoo
import gymnasium
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.test import parallel_api_test

def env(render_mode=None, nprey=1, npredator=1, dim=5, vision=2, mode='mixed', stay=True,
            moving_prey=False, learning_prey=False, comm_range=0): #, enemy_comm=False
    """
    Wrapper function for the Predator-Prey environment

        Parameters:
            render_mode (str or NoneType) -- Way in which the environment is rendered (None (default) or 'human')

        Returns:
            env (AECEnv) -- Instance of the Predator-Prey environment
    """
    if render_mode == 'human':
        internal_render_mode = render_mode
    elif not render_mode:
        internal_render_mode = None
    else:
        raise ValueError("Only the human render_mode is available.")
    env = PredatorPreyEnv(render_mode=internal_render_mode, nprey=nprey, npredator=npredator, dim=dim, vision=vision, mode=mode, stay=stay,
                          moving_prey=moving_prey, learning_prey=learning_prey, comm_range=comm_range) #, enemy_comm=enemy_comm

    # Error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)

    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None, nprey=1, npredator=1, dim=5, vision=2, mode='mixed', stay=True,
            moving_prey=False, learning_prey=False, comm_range=0): #, enemy_comm=False
    """
    Secondary wrapper function needed to support the AEC API for parallel environments

        Parameters:
            render_mode (str or NoneType) -- Way in which the environment is rendered (None (default) or 'human')

        Returns:
            env (AECEnv) -- Instance of the Predator-Prey environment
    """
    if render_mode == 'human':
        render_mode = render_mode
    elif not render_mode:
        render_mode = None
    else:
        raise ValueError("Only the human render_mode is available.")
    env = PredatorPreyEnv(render_mode=render_mode, nprey=nprey, npredator=npredator, dim=dim, vision=vision, mode=mode, stay=stay,
                          moving_prey=moving_prey, learning_prey=learning_prey, comm_range=comm_range) #, enemy_comm=enemy_comm
    env = parallel_to_aec(env)
    return env

class PredatorPreyEnv(ParallelEnv):
    metadata = {'render.modes': ['human'], 'name': 'PredatorPrey_v1'}

    def init_args(self, parser):
        """
        Additional (command line) arguments specific to this environment
        Not used in PettingZoo, must be used explicitly in the main file.
            If you don't want to use this, you can pass the arguments directly.

            Parameters:
                parser (argparse.ArgumentParser) -- Argument parser used in the main file
        """
        env = parser.add_argument_group('Prey Predator task')
        env.add_argument('--nenemies', type=int, default=1,
                         help="Total number of preys in play")
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box")
        env.add_argument('--vision', type=int, default=2,
                         help="Vision of agents")
        parser.add_argument('--mode', default='mixed', type=str,
                        help='cooperative|competitive|mixed (default: mixed)')
        env.add_argument('--stay', action="store_false", default=True,
                         help="Whether predators have an action to stay in place. Note: this replaces 'no_stay' in previous version which had the opposite behaviour")
        env.add_argument('--moving_prey', action="store_true", default=False,
                         help="Whether prey can move")
        env.add_argument('--learning_prey', action="store_true", default=False,
                         help="Whether prey can learn their own policies")
        env.add_argument('--comm_range', type=int, default=0,
                         help="Range over which agents can maintain communication. If 0, there is no limit.")
        # This was included in the previous version but never did anything, removed until implemented
        # env.add_argument('--enemy_comm', action="store_true", default=False,
        #                  help="Whether prey can communicate")

        return None

    def __init__(self, args=None, render_mode=None, nprey=1, npredator=1,
                 dim=5, vision=2, mode='mixed', stay=True,
                 moving_prey=False, learning_prey=False, comm_range=0): #, enemy_comm=False
        """
        Initialise the Predator-Prey environment
        Note: Currently defaults to using the arguments from the args without checking direct inputs
            TODO: Allow different parameters for predators and prey (perhaps also heterogeneous predators), e.g. vision
            TODO: Allow reward values as optional arguments
            TODO: Allow separate systems for prey and predator communication rather than giving all agents access to all communication

        Version 1.1.0 changed the value for agents in self.infos into a dict {"loc": np.ndarray, "alive": int}.

        Parameters:
            args (argparse.Namespace object or NoneType) -- List of arguments passed from the command line to the main file
            render_mode (str or NoneType) -- Way in which the environment is rendered (None (default) or 'human')
            nprey (int) -- Total number of preys in play (default 1)
            npredator (int) --  (default 1)
            dim (int) -- Dimension of box (default 5)
            vision (int) -- Vision of agents (default 2)
            mode (int) -- cooperative|competitive|mixed (default 'mixed')
            stay (bool) -- Whether predators have an action to stay in place. Note: this replaces 'no_stay' in previous version which had the opposite behaviour (default True)
            moving_prey (bool) -- Whether prey can move (default False)
            learning_prey (bool) -- Whether prey can learn their own policies (default False)
            comm_range (int) -- Range over which agents can maintain communication. If 0, there is no limit. (default 0)
        """
        self.__version__ = "1.1.0"

        # These parameters are consistent with previous versions
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.DEACTIVATE_PENALTY = -2.

        # Collect attributes from args or direct input
        if args:
            # Collect from args
            # General variables defining the environment : CONFIG
            params = ['dim', 'vision', 'mode', 'stay', 'moving_prey', 'learning_prey', 'comm_range'] #, 'enemy_comm'
            for key in params:
                setattr(self, key, getattr(args, key))

            self.nprey = args.nenemies
            self.npredator = args.nfriendly
        else:
            self.dim = dim
            self.vision = vision
            self.mode = mode
            self.stay = stay
            self.moving_prey = moving_prey
            self.learning_prey = learning_prey
            self.comm_range = comm_range
            # self.enemy_comm = enemy_comm

            self.nprey = nprey
            self.npredator = npredator

        self.dims = (self.dim, self.dim)

        if self.learning_prey and not self.moving_prey:
            gymnasium.logger.warn(
                "You set prey to learn policies without allowing them to move. Ignoring this setting and allowing them to move, regardless."
            )
            # No point learning a policy if the prey can't move
            self.moving_prey = True

        self.BASE = (self.dims[0] * self.dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
        #          predator + prey + grid + outside

        # Predators and prey are stored as tuple with names and int indices
        #   for differentiation and so that they can be called individually at runtime
        self.possible_predators = ['predator_' + str(i) for i in range(self.npredator)]
        self.possible_prey = ['prey_' + str(i) for i in range(self.nprey)]
        if self.learning_prey:
            self.possible_agents = self.possible_predators + self.possible_prey
        else:
            # Prey aren't agents if they aren't learning
            self.possible_agents = self.possible_predators

        # Create a mapping between agent name and id
        self.agent_name_mapping = self._create_agent_name_mapping()

        # This is a hack because the pettingzoo parallel_api_test is asking for an agent_selection
        self.agent_selection = None

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        if self.stay:
            self.naction = 5
        else:
            self.naction = 4

        # Define action and observation spaces (using a dictionary of action spaces by agent for PettingZoo)
        self._action_spaces = {agent: spaces.Discrete(self.naction) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1)
                              , dtype=int) for agent in self.possible_agents
        }
        self.observation_dim = self.vocab_size * ((2 * self.vision) + 1) * ((2 * self.vision) + 1)

        # Set up rendering
        self.render_mode = render_mode
        if render_mode == 'human':
            self.init_curses()

        return None

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        """Collect the action space for an agent
        If the agent is not specified, return the action space for the first agent"""
        if agent == None:
            agent = self.possible_agents[0]
        return self._action_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        """Collect the observation space for an agent
        If the agent is not specified, return the observation space for the first agent"""
        if agent == None:
            agent = self.possible_agents[0]
        return self._observation_spaces[agent]

    def init_curses(self):
        """Initialise human rendering using the curses library"""
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_MAGENTA, -1) #using magenta rather than red for improved colour-blind visibility
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)

        return None

    def render(self):
        """Render the environment state using the curses library"""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return None

        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for i in range(self.npredator):
            p = self.locs[i]
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'X'
            else:
                grid[p[0]][p[1]] = 'X'

        for i in range(self.nprey):
            # Skip dead prey
            if not self.active_prey[i]: continue

            p = self.locs[i+self.npredator]
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'X' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

        return None

    def close(self):
        """Release the graphical display once the environment is no longer needed"""
        if self.render_mode == 'human':
            curses.endwin()
        return None

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and return an initial observation.

            Parameters:
                seed (int) -- Random seed (default None uses an arbitrary seed)
                options -- Not used, kept for compatibility with the AEC/Pettingzoo API

            Returns:
                all_obs (dict(agent: observation object)) -- The initial joint observation of the space.
                infos (dict) -- Information about each agent
        """
        # Reset the random seed
        #   Including generating a new seed using PRNG if not provided
        self.np_random, seed = seeding.np_random(seed)

        # (Re)Initialise key attributes
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # self.episode_over = False
        self.reached_prey = np.zeros(self.npredator, dtype=int) # Whether each predator is on the target prey
        self.active_prey = np.full(self.nprey, True) # Whether each prey is active (not dead)
        self.n_living_prey = self.nprey # Number of active prey
        self.trapped_prey = np.zeros(self.nprey, dtype=int) # Number of prey with a predator on them

        # Locations
        self.locs = self._init_coordinates()

        self._set_grid()

        # stats - like success ratio
        self.stat = dict()

        # Observation is N * 2*vision+1 * 2*vision+1 * vocab_size
        #   N is the number of learning agents (possibly including prey)
        #   2*vision+1 means that agents have a visual range of self.vision on all sides
        #   vocab_size includes a one_hot encoding of all possible obsjects, size=(dims[0]*dims[1])+4
        self.empty_observation = np.zeros([2*self.vision+1, 2*self.vision+1, self.vocab_size], dtype=int)
        self.empty_observation[:,:,26] += 1 #All it sees is 'OUTSIDE_CLASS'
        self.all_obs = self._get_obs()

        # Set up agent infos with their status and locations
        self.infos = {
            agent: {"alive": 1, "loc": self.locs[self.agent_name_mapping[agent]]} for agent in self.agents
        }

        return self.all_obs, self.infos

    def _create_agent_name_mapping(self):
        """
        Create a mapping between agent name and id for self.possible_agents

            Returns:
                agent_name_mapping (dict(agent_name: agent_id)) -- Mapping between agent name and id
        """
        return dict(zip(self.possible_agents, range(len(self.possible_agents))))


    def _get_obs(self):
        """
        Collect observations for all agents

            Returns:
                all_obs (dict(agent: observation object)) -- Joint agent observations
        """
        # Representation of the full state
        self.bool_state = self.empty_bool_base_grid.copy()

        # Place all predators and prey in the state
        for i in range(self.npredator):
            p = self.locs[i]
            self.bool_state[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1

        for i in range(self.nprey):
            # Skip dead prey
            if not self.active_prey[i]:
                continue

            p = self.locs[i+self.npredator]
            self.bool_state[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1

        # Collect partial observations for all agents (possibly including prey)
        observations = {}
        for agent in self.possible_agents:
            agent_id = self.agent_name_mapping[agent]
            if agent_id >= self.npredator and not self.active_prey[agent_id - self.npredator]:
                # Dead prey receive an empty observation
                observations[agent] = self.empty_observation.copy()
            else:
                agent_loc = self.locs[agent_id]
                slice_y = slice(agent_loc[0], agent_loc[0] + (2 * self.vision) + 1)
                slice_x = slice(agent_loc[1], agent_loc[1] + (2 * self.vision) + 1)
                observations[agent] = self.bool_state[slice_y, slice_x]

        return observations

    def step(self, actions):
        """
        The agents take a step in the environment.

            Parameters:
                actions (array-like 1D) -- array-like actions chosen by the agents

            Returns:
                observations (dict(agent: observation object)) -- Joint agent observations
                rewards (dict(agent: float)) -- Rewards from the environment
                terminations (dict(agent: bool)) -- Flags denoting whether agents are taken out of action before the end of the episode
                truncations (dict(agent: bool)) -- Flags denoting whether the episode has been ended early
                infos (dict) -- Diagnostic information useful for debugging
                    The values are either agent location or a str saying that the agent is dead)
        """
        if not actions:
            gymnasium.logger.warn(
                "You are calling the step method without specifying any actions."
            )
            return {}, {}, {}, {}, {}

        # Loop through agents and actions
        for agent, act in actions.items():
            # Skip dead prey
            agent_id = self.agent_name_mapping[agent]
            if agent_id >= self.npredator and not self.active_prey[agent_id - self.npredator]:
                self.infos[agent]["alive"] = 0
                continue

            assert act <= self.naction, "Actions should be in the range [0,naction)."
            self._take_action(agent, act)

            self.infos[agent]["loc"] = self.locs[agent_id]

        if self.moving_prey and not self.learning_prey:
            for i in range(self.nprey):
                act = self.np_random.randint(self.naction)
                agent = 'random_prey_' + str(i)
                self.agent_name_mapping[agent] = i + self.npredator
                self._take_action(agent, act)

        return self._get_obs(), self._get_rewards(), self.terminations, self.truncations, self.infos

    def _take_action(self, agent, act):
        """
        Allow a single agent to take an action in the environment, moving in one of the four cardinal directions or, optional, staying in place.

            Parameters:
                agent (agent object) -- Agent to act
                act (int) -- Action to be taken, encoded into the range [0,naction)
        """
        idx = self.agent_name_mapping[agent]
        if idx < self.npredator:
            # Predators stop moving once they've reached the prey
            if self.reached_prey[idx] == 1:
                return None
        else:
            if not self.moving_prey:
                return None
            else:
                # Prey stop moving once trapped (reached) by a predator
                prey_loc = self.locs[idx]
                if np.any(np.all(self.locs[:self.npredator] == prey_loc, axis=1)):
                    self.trapped_prey[idx-self.npredator] = 1
                    return None

        # Make sure the action is valid
        assert act <= self.naction, "Actions should be in the range [0,naction)."

        # UP
        if act==0:
            if self.grid[max(0,self.locs[idx][0] + self.vision - 1),
                        self.locs[idx][1] + self.vision] != self.OUTSIDE_CLASS:
                self.locs[idx][0] = max(0, self.locs[idx][0]-1)

        # RIGHT
        elif act==1:
            if self.grid[self.locs[idx][0] + self.vision,
                        min(self.dims[1] -1, self.locs[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
                self.locs[idx][1] = min(self.dims[1]-1, self.locs[idx][1]+1)

        # DOWN
        elif act==2:
            if self.grid[min(self.dims[0]-1, self.locs[idx][0] + self.vision + 1),
                        self.locs[idx][1] + self.vision] != self.OUTSIDE_CLASS:
                self.locs[idx][0] = min(self.dims[0]-1, self.locs[idx][0]+1)

        # LEFT
        elif act==3:
            if self.grid[self.locs[idx][0] + self.vision,
                        max(0, self.locs[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
                self.locs[idx][1] = max(0, self.locs[idx][1]-1)

        # STAY
        elif act==4:
            return None

        else:
            raise ValueError(f"No action has been taken for agent {agent} with chosen action {act}")

        return None

    def _get_rewards(self):
        """
        Calculate rewards for all agents for this timestep
        For prey:
            -1*self.TIMESTEP_PENALTY if they have not been caught
            self.TIMESTEP_PENALTY if there is at least 1 predator on them
            self.DEACTIVATE_PENALTY if all predators catch them and they are killed, received max once per episode
        For predators:
            self.TIMESTEP_PENALTY if they are not at the same location as a prey (have not caught a prey)
            If they have caught a prey (are at the same location as a prey):
                cooperative mode - self.POS_PREY_REWARD * number of predators on the a prey
                competitive mode - self.POS_PREY_REWARD / number of predators on the same prey
                mixed mode - self.PREY_REWARD but the largest group of predators that are on a captured prey remain still
            When there are multiple prey, there is an additional reward for predators on the 'most caught' prey
                to promote focusing on one prey at a time.

            Returns:
                rewards (dict(agent: float)) -- Rewards from the environment (see above for logic)
        """
        nagents = len(self.possible_agents)
        reward = np.full(nagents, self.TIMESTEP_PENALTY)

        # Determine which predators have caught which prey
        n_predator_on_prey = np.zeros([self.npredator, self.nprey], dtype=int)
        for i in range(self.npredator):
            predator_loc = self.locs[i]
            for j in range(self.nprey):
                prey_loc = self.locs[j+self.npredator]
                if np.all(predator_loc == prey_loc):
                    n_predator_on_prey[i, j] += 1
        self.trapped_prey = np.any(n_predator_on_prey, axis=0)

        if self.nprey == 1:
            on_prey = n_predator_on_prey.squeeze(axis=1)
            on_prey_count = np.sum(on_prey)

            self.reached_prey[on_prey == 1] = 1

            # Rewards for predators
            if self.mode == 'cooperative':
                reward[:self.npredator][on_prey == 1] = self.POS_PREY_REWARD * on_prey_count
            elif self.mode == 'competitive':
                if on_prey_count:
                    reward[:self.npredator][on_prey == 1] = self.POS_PREY_REWARD / on_prey_count
            elif self.mode == 'mixed':
                reward[:self.npredator][on_prey == 1] = self.PREY_REWARD

                # If all predators have caught a prey, terminate the episode
                if np.all(self.reached_prey == 1):
                    self.active_prey[0] = False
                    self.n_living_prey -= 1
                    self.terminations = {k: True for k in self.terminations}
                    self.agents = {}
            else:
                raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

            # Rewards for prey
            if self.learning_prey:
                if on_prey_count == 0:
                    reward[self.npredator] = -1 * self.TIMESTEP_PENALTY
                elif np.all(self.reached_prey == 1):
                    # Large one-time penalty when caught and killed
                    reward[self.npredator] = self.DEACTIVATE_PENALTY
                else:
                    reward[self.npredator] = self.TIMESTEP_PENALTY
        elif self.nprey > 1:
            # Determine the prey with the most predators on it
            on_prey_count = np.sum(n_predator_on_prey, axis=0)
            most_caught_prey = np.argmax(on_prey_count)

            # The current most-caught prey becomes the temporary goal
            self.reached_prey[n_predator_on_prey[:, most_caught_prey] == 1] = 1

            # Rewards for predators (as for 1 prey but multiply by the number of prey a predator has caught at once)
            if self.mode == 'cooperative':
                # Reward every predator a little bit for every predator on any prey
                reward[:self.npredator] = np.sum(n_predator_on_prey * on_prey_count * self.POS_PREY_REWARD, axis=1)

                # Double reward for the most caught prey
                reward[:self.npredator] += on_prey_count[most_caught_prey] * self.POS_PREY_REWARD

                # Timestep penalty if not on any prey
                reward[:self.npredator][self.reached_prey == 0] = self.TIMESTEP_PENALTY
            elif self.mode == 'competitive':
                # Reward a predator for being on any prey
                reward[:self.npredator] = 0
                for j in range(self.nprey):
                    if on_prey_count[j] != 0:
                        reward[:self.npredator] += n_predator_on_prey[:, j] * self.POS_PREY_REWARD / on_prey_count[j]

                # I don't think that double rewards make sense in the competitive mode
                # # Double reward for being on the most caught prey
                # reward[:self.npredator][self.reached_prey == 1] += on_prey_count[most_caught_prey] / self.POS_PREY_REWARD

                # Timestep penalty if not on any prey
                reward[:self.npredator][self.reached_prey == 0] = self.TIMESTEP_PENALTY
            elif self.mode == 'mixed':
                # Reward a predator for being on any prey (different value for mixed vs other types)
                reward[:self.npredator] = np.sum(n_predator_on_prey * self.PREY_REWARD, axis=1)

                # Double reward for being on the most caught prey
                reward[:self.npredator][self.reached_prey == 1] += on_prey_count[most_caught_prey] * self.PREY_REWARD

                # Timestep penalty if not on any prey
                reward[:self.npredator][self.reached_prey == 0] = self.TIMESTEP_PENALTY

                # If all predators have caught a prey, terminate it and either end the episode or move on
                if np.all(self.reached_prey == 1):
                    # Loop through caught prey (at least including most_caught prey)
                    caught_list = np.where(on_prey_count == self.npredator)[0]
                    for caught_prey in np.flip(caught_list):
                        # Remove caught prey from various attributes (noting that indexing may be messed up by removing previous prey)
                        self.n_living_prey -= 1

                        # Change the loc to one of the outside spots
                        #   it gets ignored by the observe functions but I don't want it interfering with the 'on_prey' stuff
                        dead_prey_idx = caught_prey
                        self.locs[self.npredator+dead_prey_idx] = [0, 0]
                        self.active_prey[dead_prey_idx] = False
                        if self.learning_prey:
                            self.all_obs[self.possible_agents[self.npredator+dead_prey_idx]] = self.empty_observation.copy()

                            # Account for the possibility of self.agents being smaller than npredator+nprey if a prey has already been caught
                            ndead = len(np.where(self.active_prey == False)) - 1
                            if caught_prey >= ndead:
                                dead_prey_idx = caught_prey - ndead

                            self.terminations[self.possible_agents[self.npredator+dead_prey_idx]] = True
                            self.agents.pop(self.possible_agents[self.npredator+dead_prey_idx])

                        # If all prey have been caught, terminate the episode
                        if self.n_living_prey == 0:
                            # All prey have been caught, terminate all agents ready to end the episode
                            self.terminations = {k: True for k in self.terminations}
                            self.agents = {}
                else:
                    # If some agents haven't caught the prey, give them the timestep penalty again
                    reward[:self.npredator][reward[:self.npredator] == 0] = self.TIMESTEP_PENALTY
            else:
                raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

            # Rewards for prey
            if self.learning_prey:
                for i in range(self.nprey):
                    # Dead prey receive a constant reward equal to having been caught
                    if not self.active_prey[i]:
                        reward[self.npredator+i] = self.TIMESTEP_PENALTY #0
                    else:
                        if on_prey_count[i] == 0:
                            reward[self.npredator+i] = -1 * self.TIMESTEP_PENALTY
                        elif on_prey_count[i] == self.npredator:
                            # Large one-time penalty when caught and killed
                            reward[self.npredator+i] = self.DEACTIVATE_PENALTY
                        else:
                            reward[self.npredator+i] = self.TIMESTEP_PENALTY #0
        else:
            raise ValueError("This environment requires at least one prey, nprey (nenemies) must be an integer >= 1")

        # Success ratio
        if self.mode != 'competitive':
            self.stat['success'] = self.nprey - self.n_living_prey

        self.rewards = {agent: reward[self.agent_name_mapping[agent]] for agent in self.possible_agents}
        return self.rewards

    def _init_coordinates(self):
        """
        Set up random initial coordinates for all predators and prey

            Returns:
                locs (np array, shape: npredator+nprey x 2) -- coordinates for the predators and prey in the environment
        """
        idx = np.random.choice(np.prod(self.dims),(self.npredator + self.nprey), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        """
        Set up an empty grid to serve as the set of possible states in the environment
            Note: The padding means that all locations are offset by self.vision.
                This is accounted for when determining observations, movement, etc.
        """
        self.grid = np.arange(self.BASE).reshape(self.dims)

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

        return None

    def _onehot_initialization(self, a):
        """
        Creates a onehot encoded representation of the environment's grid

            Parameters:
                a (grid object) -- A grid to encode

            Returns:
                out (numpy array) -- A onehot encoded representation of the environment's grid
        """
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        """
        Onehot encode an array based on the values at each index

            Parameters:
                idx (numpy array) -- Array to be encoded
                axis (int) -- Axis along which to insert the encoded values

            Returns:
                out (tuple) -- Tuple of indices where the value of the onehot encoding will be 1
        """
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def reward_terminal(self):
        """Return a zero reward when the environment is terminal
        Note: This is adapted from Predator-Prey v0, it may not be necessary for you to use it."""
        if self.terminations[self.possible_agents[0]] or self.truncations[self.possible_agents[0]]:
            self.rewards = {agent: 0 for agent in self.possible_agents}
        else:
            self.rewards = self._get_rewards()
        return self.rewards

    def get_comm_range_mask(self):
        """
        Determine which agents are in range to communicate and mask off communication otherwise.
            Note: Communication ranges are always square (like vision).
            Note: We do not differentiate predators and prey if both are able to communicate.
            Note: Agents can always communicate with themselves.

            Returns:
                comm_range_mask (2D numpy array) -- A mask to apply to a square adjacency matrix
                    for multi-agent reinforcement learning methods with communication
        """
        nagents = len(self.possible_agents)

        # If comm range is unlimited, return a trivial mask
        if self.comm_range == 0:
            return np.ones([nagents, nagents])

        comm_range_mask = np.zeros([nagents, nagents])

        # Agents will never be out of range of themselves
        comm_range_mask += np.eye(nagents)

        # Main loop for determining whether agents are in range
        for i in range(nagents):
            # If i is a dead prey, mask all communication
            if i > self.npredator:
                if not self.active_prey[i-self.npredator]:
                    comm_range_mask[i, i] = 0
                    continue

            for j in range(i+1, nagents):
                # If j is a dead prey, mask communication
                if j > self.npredator:
                    if not self.active_prey[j-self.npredator]:
                        continue

                y_dist = np.abs(self.locs[i][0] - self.locs[j][0])
                x_dist = np.abs(self.locs[i][1] - self.locs[j][1])
                if np.max(y_dist, x_dist) <= self.comm_range:
                    comm_range_mask[i, j] = 1
                    comm_range_mask[j, i] = 1

        return comm_range_mask

    def get_stat(self):
        """Collect self.stat from this environment"""
        if hasattr(self, 'stat'):
            self.stat.pop('steps_taken', None)
            return self.stat
        else:
            return dict()


if __name__ == "__main__":
    env = PredatorPreyEnv()
    parallel_api_test(env, num_cycles=1_000_000)
