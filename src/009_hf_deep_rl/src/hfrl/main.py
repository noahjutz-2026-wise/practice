import gymnasium as gym
from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def main():
    env = gym.make("LunarLander-v3")
    model = PPO("MlpPolicy", env, verbose=1)
    _ = model.learn(total_timesteps=int(2e5))

    eval_env = Monitor(gym.make("LunarLander-v3", render_mode="human"))

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    print("packaging")

    _ = package_to_hub(
        model=model,
        model_name="ppo_lunarlander_v3",
        model_architecture="PPO",
        env_id="LunarLander-v3",
        eval_env=eval_env,
        repo_id="noahjutz/ppo_lunarlander_3",
        commit_message="commit",
    )
