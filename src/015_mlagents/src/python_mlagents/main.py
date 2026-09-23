from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import PPO


def main():
    unity_env = UnityEnvironment("/home/noah/Downloads/export/unitybuild.x86_64")
    env = UnityToGymWrapper(unity_env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    print("Saving model to unity_model.zip")
    model.save("unity_model")


if __name__ == "__main__":
    main()
