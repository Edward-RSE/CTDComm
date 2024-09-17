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

# Gymnasium and PettingZoo
import gymnasium
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

def env(render_mode=None, render_mode=None, dim=None, vision=None, moving_prey=None, enemy_comm=None, nprey=None, npredator=None, stay=None, comm_range=None):
    """
    Wrapper function for the Predator-Prey environment

        Parameters:
            render_mode (str or NoneType) -- Way in which the environment is rendered (None (default) or 'human')

        Returns:
            env (AECEnv) -- Instance of the Predator-Prey environment
    """
    if render_mode == 'human':
        internal_render_mode = render_mode
    else:
        raise ValueError("Only the human render_mode is available.")
    env = PredatorPreyEnv(render_mode=internal_render_mode, dim=dim, vision=vision, moving_prey=moving_prey, enemy_comm=enemy_comm,
                          nprey=nprey, npredator=npredator, stay=stay, comm_range=comm_range)

    # Error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)

    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None, render_mode=None, dim=None, vision=None, moving_prey=None, enemy_comm=None, nprey=None, npredator=None, stay=None, comm_range=None):
    """
    Secondary wrapper function needed to support the AEC API for parallel environments

        Parameters:
            render_mode (str or NoneType) -- Way in which the environment is rendered (None (default) or 'human')

        Returns:
            env (AECEnv) -- Instance of the Predator-Prey environment
    """
    env = PredatorPreyEnv(render_mode=render_mode, dim=dim, vision=vision, moving_prey=moving_prey, enemy_comm=enemy_comm,
                          nprey=nprey, npredator=npredator, stay=stay, comm_range=comm_range)
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
                         help="Vision of predator")
        env.add_argument('--moving_prey', action="store_true", default=False,
                         help="Whether prey can move")
        env.add_argument('--learning_prey', action="store_true", default=False,
                         help="Whether prey can learn their own policies")
        env.add_argument('--enemy_comm', action="store_true", default=False,
                         help="Whether prey can communicate")
        env.add_argument('--no_stay', action="store_true", default=False,
                         help="Whether predators have an action to stay in place")
        parser.add_argument('--mode', default='mixed', type=str,
                        help='cooperative|competitive|mixed (default: mixed)')
        env.add_argument('--comm_range', type=int, default=0,
                         help="Range over which agents can maintain communication. If 0, there is no limit.")
        
        return None

    def __init__(self, args, render_mode=None, dim=None, vision=None, moving_prey=None, enemy_comm=None, nprey=None, npredator=None, stay=None, comm_range=None):
        """
        Initialise the Predator-Prey environment
        Note: Currently defaults to using the arguments from the args without checking direct inputs
            TODO: Handle attributes being specified in args AND as direct inputs to make()/__init__()

        Parameters:
            render_mode (str or NoneType) -- Way in which the environment is rendered (None (default) or 'human')
        """
        self.__version__ = "1.0.1"

        # These parameters are consistent with previous versions
        # TODO: allow reward values as optional arguments
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05

        # Collect attributes from args or direct input
        if args:
            # Collect from args
            # General variables defining the environment : CONFIG
            params = ['dim', 'vision', 'moving_prey', 'enemy_comm']
            for key in params:
                setattr(self, key, getattr(args, key))
            
            self.nprey = args.nenemies
            self.npredator = args.nfriendly
            self.stay = not args.no_stay
            self.comm_range = args.comm_range
        else:
            self.dim = dim
            self.vision = vision
            self.moving_prey = moving_prey
            self.enemy_comm = enemy_comm

            self.nprey = nprey
            self.npredator = npredator
            self.stay = stay
            self.comm_range = comm_range
        self.mode = render_mode
        self.dims = (self.dim, self.dim)

        self.learning_prey = args.learning_prey
        if self.learning_prey:
            # No point learning a policy if the prey can't move
            self.moving_prey = True
        else:
            self.moving_prey = args.moving_prey

        self.BASE = (self.dims[0] * self.dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
        #          predator + prey + grid + outside

        # Predators and prey are stored as tuple with names and int indices
        #   for differentiation and so that they can be called individually at runtime
        self.possible_predators = [('predator_' + str(i), i) for i in range(self.npredator)]
        self.possible_prey = [('prey_' + str(i), i+self.npredator) for i in range(self.nprey)]
        if self.learning_prey:
            self.possible_agents = self.possible_predators + self.possible_prey
        else:
            # Prey aren't agents if they aren't learning
            self.possible_agents = self.possible_predators

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
        super().reset(seed=seed)
        
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

        # stat - like success ratio
        self.stat = dict()

        # Observation is N * 2*vision+1 * 2*vision+1
        #   N is the number of learning agents (possibly including prey)
        #   2*vision+1 means that agents have a visual range of self.vision on all sides
        self.empty_observation = np.ones([2*self.vision+1, 2*self.vision+1], dtype=int)*26
        self.all_obs = {}
        self.all_obs = self._get_obs()

        # Set up agent infos with their locations
        self.infos = {agent: self.locs[agent[1]] for agent in self.agents}

        return self.all_obs, self.infos
    
    def _get_obs(self):
        """
        Collect observations for all agents
        
            Returns:
                all_obs (dict(agent: observation object)) -- Joint agent observations
        """
        for agent in self.possible_agents:
            # Dead prey receive an empty observation
            if agent[1] >= self.npredator:
                if not self.active_prey[agent[1]-self.npredator]:
                    self.all_obs[agent] = self.empty_observation.copy()
            else:
                self.all_obs[agent] = self._observe(agent)
        return self.all_obs
    
    def _observe(self, agent):
        """
        Collect the partial observation of a single agent
        Note: agent positions are stored as [y-self.vision, x-self.vision],
            where x and y are the actual coordinates in the environment

            Parameters:
                agent (agent object) -- Agent for which to collect the observation
            
            Returns:
                obs_grid (observation object) -- Partial observation for the agent
        """
        # Get the location of the observing agent
        agent_loc = self.locs[agent[1]]
        range_y = np.arange(agent_loc[0], agent_loc[0] + (2 * self.vision) + 2)
        range_x = np.arange(agent_loc[1], agent_loc[1] + (2 * self.vision) + 2)

        # Get the region within the agent's observation range
        bool_base_grid = self.empty_bool_base_grid.copy()
        slice_y = slice(range_y[0], range_y[-1]+1)
        slice_x = slice(range_x[0], range_x[-1]+1)
        obs_grid = bool_base_grid[slice_y, slice_x]

        # Add the predators and prey in the agent's observation range
        for i in range(self.npredator):
            p = self.locs(i)
            if p[0]+self.vision in range_y:
                if p[1]+self.vision in range_x:
                    obs_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1
        
        for i in range(self.nprey):
            # Skip dead prey
            if not self.active_prey[i]: continue
            
            p = self.locs(i+self.npredator)
            if p[0]+self.vision in range_y:
                if p[1]+self.vision in range_x:
                    obs_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1

        self.all_obs[agent[1]] = obs_grid
        return obs_grid

    def step(self, actions):
        """
        The agents take a step in the environment.

            Parameters:
                actions (array-like 1D) -- array-like actions chosen by the agents

            Returns:
                observations (dict(agent: observation object)) -- Joint agent observations
                rewards (dict(agent: float)) -- Rewards from the environment
                terminations (dict(agent: bool)) -- Flags denoting whether the episode has been ended early
                truncations (dict(agent: bool)) -- Flags denoting whether agents are taken out of action before the end of the episode
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
            if agent[1] >= self.npredator and not self.active_prey[agent[1]-self.npredator]:
                self.infos[agent] = "Dead prey"
                continue
            
            assert act <= self.naction, "Actions should be in the range [0,naction)."
            self._take_action(agent, act)

            self.infos[agent] = self.locs[agent[1]]
        
        if self.moving_prey and not self.learning_prey:
            for i in range(self.nprey):
                act = self.np_random.integers(self.naction)
                agent = ('random_prey_'+str(i), i+self.npredator)
                self._take_action(agent, act)

        return self._get_obs(), self._get_rewards(), self.terminations, self.truncations, self.infos

    def _take_action(self, agent, act):
        """
        Allow a single agent to take an action in the environment, moving in one of the four cardinal directions or, optional, staying in place.

            Parameters:
                agent (agent object) -- Agent to act
                act (int) -- Action to be taken, encoded into the range [0,naction)
        """
        idx = agent[1]
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
        if act==0 and self.grid[max(0,
                                self.locs[idx][0] + self.vision - 1),
                                self.locs[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.locs[idx][0] = max(0, self.locs[idx][0]-1)

        # RIGHT
        elif act==1 and self.grid[self.locs[idx][0] + self.vision,
                                min(self.dims[1] -1,
                                    self.locs[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.locs[idx][1] = min(self.dims[1]-1,
                                            self.locs[idx][1]+1)

        # DOWN
        elif act==2 and self.grid[min(self.dims[0]-1,
                                    self.locs[idx][0] + self.vision + 1),
                                    self.locs[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.locs[idx][0] = min(self.dims[0]-1,
                                            self.locs[idx][0]+1)

        # LEFT
        elif act==3 and self.grid[self.locs[idx][0] + self.vision,
                                    max(0,
                                    self.locs[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
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
        For predators:
            self.TIMESTEP_PENALTY if they are not at the same location as a prey (have not caught a prey)
            If they have caught a prey (are at the same location as a prey):
                self.POS_PREY_REWARD * number of predators on the a prey - cooperative mode
                self.TIMESTEP_PENALTY or self.POS_PREY_REWARD / number of predators on the same prey - competitive mode
                self.PREY_REWARD but the largest group of predators that are on a captured prey remain still - mixed mode
        
            Returns:
                rewards (dict(agent: float)) -- Rewards from the environment (see above for logic)
        """
        nagents = len(self.possible_agents)
        reward = np.full(nagents, self.TIMESTEP_PENALTY)

        # Determine which predators have caught which prey
        n_predator_on_prey = np.zeros([self.npredator, self.nprey], dtype=int)
        for i in range(self.npredator):
            predator_loc = self.locs(i)
            for j in range(self.nprey):
                prey_loc = self.locs(j+self.npredator)
                if np.all(predator_loc == prey_loc):
                    n_predator_on_prey[i, j] += 1
        self.trapped_prey = np.any(n_predator_on_prey, axis=0)
        
        if self.nprey == 1:
            on_prey = n_predator_on_prey.squeeze(axis=1)
            on_prey_count = np.sum(on_prey)

            self.reached_prey[on_prey] = 1

            # Rewards for predators
            if self.mode == 'cooperative':
                reward[on_prey] = self.POS_PREY_REWARD * on_prey_count
            elif self.mode == 'competitive':
                if on_prey_count:
                    reward[on_prey] = self.POS_PREY_REWARD / on_prey_count
            elif self.mode == 'mixed':
                reward[on_prey] = self.PREY_REWARD

                # If all predators have caught a prey, truncate the episode
                if np.all(self.reached_prey == 1):
                    self.active_prey[0] = False
                    self.n_living_prey -= 1
                    self.truncations = {k: True for k in self.truncations}
                    self.all_obs[self.possible_agents[dead_prey_idx]] = self.empty_observation.copy()
            else:
                raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

            # Rewards for prey
            if on_prey_count == 0:
                reward[self.npredator] = -1 * self.TIMESTEP_PENALTY
            else:
                reward[self.npredator:] = 0
        elif self.nprey > 1:
            # Determine the prey with the most predators on it
            on_prey_count = np.sum(n_predator_on_prey, axis=0)
            most_caught_prey = np.argmax(on_prey_count)

            # The current most-caught prey becomes the temporary goal
            self.reached_prey[n_predator_on_prey[:, most_caught_prey]] = 1

            # Rewards for predators (as for 1 prey but multiply by the number of prey a predator has caught at once)
            if self.mode == 'cooperative':
                reward = np.sum(n_predator_on_prey * self.POS_PREY_REWARD * on_prey_count, axis=1)
            elif self.mode == 'competitive':
                for j in range(self.nprey):
                    if on_prey_count[j] != 0:
                        reward = n_predator_on_prey[:, j] * self.POS_PREY_REWARD / on_prey_count[j]
            elif self.mode == 'mixed':
                reward = np.sum(n_predator_on_prey * self.PREY_REWARD, axis=1)

                # If all predators have caught a prey, truncate it and either end the episode or move on
                if np.all(self.reached_prey == 1):
                    # Loop through caught prey (at least including most_caught prey)
                    caught_list = np.where(on_prey_count == self.npredator)[0]
                    for caught_prey in np.flip(caught_list):
                        # Remove caught prey from various attributes (noting that indexing may be messed up by removing previous prey)
                        self.n_living_prey -= 1
                        if self.n_living_prey == 0:
                            # All prey have been caught, truncate all agents ready to end the episode
                            self.truncations = {k: True for k in self.truncations}

                        # Change the loc to one of the outside spots
                        #   it gets ignored by the observe functions but I don't want it interfering with the 'on_prey' stuff
                        dead_prey_idx = caught_prey
                        self.locs[self.npredator+caught_prey] = [0, 0]
                        self.active_prey[caught_prey] = False
                        self.all_obs[self.possible_agents[dead_prey_idx]] = self.empty_observation.copy()
                        
                        # Account for the possibility of self.agents being smaller than npredator+nprey if a prey has already been caught
                        ndead = len(np.where(self.active_prey == False)) - 1
                        if caught_prey >= ndead:
                            dead_prey_idx = caught_prey - ndead

                        self.truncations[self.agents[self.npredator+dead_prey_idx]] = True
            else:
                raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

            # Rewards for prey
            for i in range(self.nprey):
                # Dead prey receive a constant reward equal to having been caught
                if not self.active_prey[i]:
                    reward[self.npredator+i] = 0
                else:
                    if on_prey_count[i] == 0:
                        reward[self.npredator+i] = -1 * self.TIMESTEP_PENALTY
                    else:
                        reward[self.npredator+i] = 0
        else:
            raise ValueError("This environment requires at least one prey, args.nenemies must be an integer >= 1")

        # Success ratio
        if self.mode != 'competitive':
            self.stat['success'] = self.nprey - self.n_living_prey

        self.rewards = {agent: reward[agent[1]] for agent in self.agents}
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
        TODO: work out if this actually gets used"""
        self.rewards = {agent: 0 for agent in self.agents}
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