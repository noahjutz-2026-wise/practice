import itertools
import math

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

unity_env = UnityEnvironment("/home/noah/Downloads/export/unitybuild.x86_64")

env = UnityToGymWrapper(unity_env)

done = False
for i in itertools.count():
    if done:
        print("done")
        r = env.reset()
        done = False
        continue
    r = env.step([math.sin(i), math.cos(i)])
    done = r[-2]


def main():
    pass
