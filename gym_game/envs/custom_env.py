import gym
from gym import spaces
import numpy as np
from gym_game.envs.pygame_2d import PyGame2D
from gym_game.envs.memory_task import MemoryTask

class RaceEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self):
        self.pygame = PyGame2D()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=np.int)

    def reset(self):
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        return obs

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.view()


class MemEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self, task_image):
        self.task_image = task_image
        self.task_env = MemoryTask(self.task_image)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.array([-10]*2025), np.array([10]*2025), dtype=np.int)

    def reset(self):
        try:
            del self.task_env
        except:
            pass
        self.task_env = MemoryTask(self.task_image)
        obs = self.task_env.observe()
        return obs

    def step(self, action):
        self.task_env.update_by_action(action)
        reward = self.task_env.evaluate()
        obs = self.task_env.observe()
        done = self.task_env.is_done()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.task_env.view()
