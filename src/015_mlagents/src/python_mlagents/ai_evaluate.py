from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from stable_baselines3 import PPO


def main():
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=20.0)
    stats_channel = StatsSideChannel()
    env = UnityToGymWrapper(
        UnityEnvironment(
            "/home/noah/Downloads/export/unitybuild.x86_64",
            no_graphics=True,
            side_channels=[channel, stats_channel],
        )
    )

    model = PPO.load("unity_model", env=env)
    successes, total = 0, 50
    for ep in range(total):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
        if reward > 2.0:
            successes += 1
        stats = stats_channel.get_and_reset_stats()
        path_ratio = stats.get("path_completion_ratio", [(0,)])[0][0]
        print(
            f"Episode {ep + 1:2d}/{total}: Goal={'Yes' if reward > 2.0 else 'No '} (Path: {path_ratio * 100:4.1f}%, Reward: {reward:5.2f})"
        )

    print(f"\nFinal Success Rate: {successes}/{total} ({successes / total * 100:.1f}%)")
    env.close()


if __name__ == "__main__":
    main()

