import gym
import numpy as np
from cenv import CEnv

env = CEnv("./test.so")

print(env.observation_space)
for i in range(100):
    obs, reward, term, trunc, info = env.step(0)

    print(reward)


