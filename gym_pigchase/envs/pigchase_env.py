import gym
from gym import spaces
import numpy.random as rnd
import numpy as np
from numpy import uint8
import logging
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import colors

# create logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PigChaseEnv(gym.Env):
    """The Pig Chase Environment"""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    
    def __init__(self, rendering=True, pause=False):
        #logger.debug("Init ColorMatchingEnv")
        self.viewer = None
        
        self.n_agents = 2
        self.max_steps = 15
        self.remaining_steps = self.max_steps
        # 5 actions per agent
        self.action_space = spaces.MultiDiscrete([[0, 4]] * self.n_agents)
        self.actions = {
            0: 'X',
            1: 'N',
            2: 'W',
            3: 'S',
            4: 'E'
        }
        
        self.states = np.zeros([2,2], dtype=uint8)
        self.pig_state = np.zeros([1,2], dtype=uint8)
        self.pits_positions = [np.array([3,0], dtype=uint8), np.array([3,6], dtype=uint8)]
        
        self.GRASS = 0
        self.AGENTS = [1,2]
        self.PIG = 3
        self.WALL = 4
        self.PIT = 5
        
        self.base_grid = np.array([[self.WALL, self.WALL, self.WALL, self.WALL, self.WALL, self.WALL, self.WALL],
                                   [self.WALL, self.GRASS, self.GRASS, self.GRASS, self.GRASS, self.GRASS, self.WALL],
                                   [self.WALL, self.GRASS, self.WALL, self.GRASS, self.WALL, self.GRASS, self.WALL],
                                   [self.PIT, self.GRASS, self.GRASS, self.GRASS, self.GRASS, self.GRASS, self.PIT],
                                   [self.WALL, self.GRASS, self.WALL, self.GRASS, self.WALL, self.GRASS, self.WALL],
                                   [self.WALL, self.GRASS, self.GRASS, self.GRASS, self.GRASS, self.GRASS, self.WALL],
                                   [self.WALL, self.WALL, self.WALL, self.WALL, self.WALL, self.WALL, self.WALL]])
        self.NE_CORNER = (1,5)
        self.NW_CORNER = (1,1)
        self.SW_CORNER = (5,1)
        self.SE_CORNER = (5,5)
        self.grid = self.base_grid
        self.spawns = self._spawns()
        
        self.pause = pause
        self.rendering = rendering
        self.col_map = colors.ListedColormap(['green', 'blue', 'red', 'yellow', 'orange', 'black'])
        
    def _spawns(self):
        x,y = np.where(self.grid == self.GRASS)
        return np.array(zip(x,y))
    
    def config_env(self,rendering=False, pause=False):
        "Needed, since there are is no way to init a gym env with arguments"
        self.__init__(rendering, pause)
    
    
    def _step(self, actions):
        """
        Take one step with given actions

        :param actions: The actions that the agents take
        :returns: obss, rewards, done, args
        """
        logger.debug("====================")
        logger.debug("New step")
        logger.debug("Remaining steps: %d" % self.remaining_steps)
        logger.debug("")
        
        self._pig_step()
        
        self.remaining_steps -= 1
        for i in xrange(self.n_agents):
            agent_action = raw_input("Agent %d action: " % i)
            # agent_action = self.actions[actions[i]]

            logger.debug("Agent:      %d" % i)
            logger.debug("Prev state: [%d, %d]" % (self.states[i,0],self.states[i,1]))
            logger.debug("Action:     %c" % agent_action)
            self._update_agent_state(i, agent_action)
            
            logger.debug("New state:  [%d, %d]" % (self.states[i][0], self.states[i][1]))
            logger.debug("")
        
        self.grid_update()
        rewards = self._compute_reward()
        #logger.debug("Rewards: [%d, %d]" % (rewards[0], rewards[1]))
        done = self.remaining_steps == 0 or self.escaped or self.pinched
        
        
        
        return self.grid.copy(), rewards, done, {}
    
    def _compute_reward(self):
        """
        Compute the reward

        :return: reward
        """
        if self._escaped():
            logger.debug('ESCAPED')
            return 25
        elif self._pinched():
            logger.debug('PINCHED')
            return 5
        else:
            return -1
        
    def _pinched(self):
        potential_state_N = tuple(self.pig_state - np.array([1,0]))
        potential_state_S = tuple(self.pig_state + np.array([1,0]))
        potential_state_E = tuple(self.pig_state + np.array([0,1]))
        potential_state_W = tuple(self.pig_state - np.array([0,1]))
        
        self.pinched =  ((self.grid[potential_state_N] == self.AGENTS[0] or
                          self.grid[potential_state_N] == self.AGENTS[1]) and
                         (self.grid[potential_state_S] == self.AGENTS[0] or
                          self.grid[potential_state_S] == self.AGENTS[1])) or \
                        ((self.grid[potential_state_E] == self.AGENTS[0] or
                          self.grid[potential_state_E] == self.AGENTS[1]) and
                         (self.grid[potential_state_W] == self.AGENTS[0] or
                          self.grid[potential_state_W] == self.AGENTS[1])) or \
                        (self.grid[self.NW_CORNER] == self.PIG and
                         (self.grid[potential_state_E] == self.AGENTS[0] or
                          self.grid[potential_state_E] == self.AGENTS[1]) and
                         (self.grid[potential_state_S] == self.AGENTS[0] or
                          self.grid[potential_state_S] == self.AGENTS[1])) or \
                        (self.grid[self.NE_CORNER] == self.PIG and
                         (self.grid[potential_state_W] == self.AGENTS[0] or
                          self.grid[potential_state_W] == self.AGENTS[1]) and
                         (self.grid[potential_state_S] == self.AGENTS[0] or
                          self.grid[potential_state_S] == self.AGENTS[1])) or \
                        (self.grid[self.SE_CORNER] == self.PIG and
                         (self.grid[potential_state_W] == self.AGENTS[0] or
                          self.grid[potential_state_W] == self.AGENTS[1]) and
                         (self.grid[potential_state_N] == self.AGENTS[0] or
                          self.grid[potential_state_N] == self.AGENTS[1])) or \
                        (self.grid[self.SW_CORNER] == self.PIG and
                         (self.grid[potential_state_E] == self.AGENTS[0] or
                          self.grid[potential_state_E] == self.AGENTS[1]) and
                         (self.grid[potential_state_N] == self.AGENTS[0] or
                          self.grid[potential_state_N] == self.AGENTS[1]))
        
        return self.pinched
    
    def _escaped(self):
        self.escaped = np.array_equal(self.pig_state, self.pits_positions[0]) \
                       or np.array_equal(self.pig_state, self.pits_positions[1])
        return self.escaped
    
    def _pig_step(self):
        """
        Updates the pig's state

        :param agent_action: agent's action
        :return: nothing
        """
        agent_action = raw_input("Pig action: ")#np.random.randint(low=0,high=5)
        
        if agent_action == 'N':
            potential_state = tuple(self.pig_state - np.array([1,0], dtype=uint8))
            if not(self.grid[potential_state] == self.AGENTS[0] or
                           self.grid[potential_state] == self.AGENTS[1] or
                           self.grid[potential_state] == self.WALL):
                logger.debug('OK')
                self.pig_state[0] -= 1
        elif agent_action == 'S':
            potential_state = tuple(self.pig_state + np.array([1, 0], dtype=uint8))
            if not (self.grid[potential_state] == self.AGENTS[0] or
                            self.grid[potential_state] == self.AGENTS[1] or
                            self.grid[potential_state] == self.WALL):
                logger.debug('OK')
                self.pig_state[0] += 1
        elif agent_action == 'W':
            potential_state = tuple(self.pig_state - np.array([0, 1], dtype=uint8))
            if not (self.grid[potential_state] == self.AGENTS[0] or
                            self.grid[potential_state] == self.AGENTS[1] or
                            self.grid[potential_state] == self.WALL):
                logger.debug('OK')
                self.pig_state[1] -= 1
        elif agent_action == 'E':
            potential_state = tuple(self.pig_state + np.array([0, 1], dtype=uint8))
            if not (self.grid[potential_state] == self.AGENTS[0] or
                            self.grid[potential_state] == self.AGENTS[1] or
                            self.grid[potential_state] == self.WALL):
                logger.debug('OK')
                self.pig_state[1] += 1
                
    def _update_agent_state(self, id, agent_action):
        """
        Updates the agent's state

        :param id: agent's id
        :param agent_action: agent's action
        :return: nothing
        """
        if agent_action == 'N':
            potential_state = tuple(self.states[id] - np.array([1,0], dtype=uint8))
            if not((potential_state == self.pig_state).all()
                   or self.grid[potential_state] == self.WALL
                   or self.grid[potential_state] == self.AGENTS[np.mod(id+1,2)]):
                logger.debug('OK')
                self.states[id,0] -= 1
        elif agent_action == 'S':
            potential_state = tuple(self.states[id] + np.array([1, 0], dtype=uint8))
            if not ((potential_state == self.pig_state).all()
                    or self.grid[potential_state] == self.WALL
                    or self.grid[potential_state] == self.AGENTS[np.mod(id+1,2)]):
                logger.debug('OK')
                self.states[id,0] += 1
        elif agent_action == 'W':
            potential_state = tuple(self.states[id] - np.array([0, 1], dtype=uint8))
            if not ((potential_state == self.pig_state).all()
                    or self.grid[potential_state] == self.WALL
                    or self.grid[potential_state] == self.PIT
                    or self.grid[potential_state] == self.AGENTS[np.mod(id + 1, 2)]):
                logger.debug('OK')
                self.states[id,1] -= 1
        elif agent_action == 'E':
            potential_state = tuple(self.states[id] + np.array([0, 1], dtype=uint8))
            if not ((potential_state == self.pig_state).all()
                    or self.grid[potential_state] == self.WALL
                    or self.grid[potential_state] == self.PIT
                    or self.grid[potential_state] == self.AGENTS[np.mod(id + 1, 2)]):
                logger.debug('OK')
                self.states[id,1] += 1
    
    
    def _reset(self):
        """
        Reset the environment

        :return: agents' states
        """
        #logger.debug("")
        #logger.debug("+++++++++++++++++++")
        #logger.debug("Reseting")
        # self._reset_colors()
        self._reset_states()
        self.remaining_steps = self.max_steps
        self._config_rendering()
        self.grid_update()
        
        #logger.debug(self.grid)
        return self.grid.copy()
    
    def grid_update(self):
        """
        Update the grid
        
        :return: nothing
        """
        self.grid = self.base_grid.copy()
        for i in xrange(self.n_agents):
            self.grid[tuple(self.states[i])] = self.AGENTS[i]

        self.grid[tuple(self.pig_state)] = self.PIG
            
    def _reset_states(self):
        """
        Reset the agents' states

        :return: nothing
        """
        for i in xrange(self.n_agents):
            self.states[i] = self.spawns[rnd.choice(len(self.spawns))]
            if(i == 1):
                while np.array_equal(self.states[0], self.states[1]):
                    self.states[i] = self.spawns[rnd.choice(len(self.spawns))]
        self.pig_state = self.spawns[rnd.choice(len(self.spawns))]
        while np.array_equal(self.pig_state, self.states[1]) or np.array_equal(self.pig_state, self.states[0]) :
            self.pig_state = self.spawns[rnd.choice(len(self.spawns))]
    
        #logger.debug("Reset state of agent %d to %s" % (i, self.states[i]))
    
    
    def _config_rendering(self):
        """
        Configure the rendering.
        Open a Figure and set the color map

        :return: nothing
        """
        if self.rendering:
            plt.close('all')
            plt.figure(1, frameon=False)
    
   
            
    def _render(self, mode='human', close=False):
        """
        Render the environment

        :param mode: rendering mode
        :param close: close the view or not
        :return: nothing
        """
        if close:
            #logger.debug("Close")
            plt.close(1)
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        #logger.debug("Rendering")
        
        if mode == 'human':
            
            plt.ion()
            plt.imshow(self.grid, interpolation='nearest', cmap=self.col_map, vmax=6, vmin=0 )
            plt.title(self.remaining_steps)
            plt.show(block=self.pause)
            plt.pause(1)
        else:
            logger.error('Unsupported rendering mode %s' % mode)
