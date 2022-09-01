import gym
import numpy as np
from cenv import CEnv

env = CEnv("./test.so")

for i in range(100):
    obs, reward, term, trunc, info = env.step(int(0))

    print(reward)


