import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from pg import *
import gym


env = gym.make('MountainCar-v0')  # 游戏环境
x = np.array([0.3, -0.02])
print(env.observation_space.high)
print(env.observation_space.low)
m = env.observation_space.high - env.observation_space.low
x1 = (x - env.observation_space.low) / m
print(x1)
