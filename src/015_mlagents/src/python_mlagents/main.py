import os
from collections import deque
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateStopCallback(BaseCallback):
    """Monitors episodes and stops training once the 75% success rate target is reached."""

    def __init__(self, target_success_rate: float = 0.75, window_size: int = 30, verbose: int = 1):
        super().__init__(verbose)
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.total_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for done, rew in zip(dones, rewards):
            if done:
                self.total_episodes += 1
                # Reaching the goal yields +5.0 reward, whereas holes/timeouts yield <= 0
                is_goal = rew > 2.0
                self.window.append(1.0 if is_goal else 0.0)
                rate = sum(self.window) / len(self.window)
                if self.total_episodes % 5 == 0 or len(self.window) == self.window_size:
                    print(
                        f"Episode {self.total_episodes}: Goal={is_goal}, "
                        f"Step Reward={rew:.2f}, Rolling Success Rate={rate * 100:.1f}% ({len(self.window)}/{self.window_size})"
                    )
                if len(self.window) >= self.window_size and rate >= self.target_success_rate:
                    print(f"\n[TARGET REACHED] {self.target_success_rate * 100:.1f}% success rate achieved ({rate * 100:.1f}%). Stopping training.")
                    return False
        return True


def load_pretrained_policy(model: PPO, onnx_path: str):
    """Loads weights from Easy.onnx into the PPO actor policy network."""
    model_onnx = onnx.load(onnx_path)
    w = {
        init.name: torch.tensor(numpy_helper.to_array(init), dtype=torch.float32)
        for init in model_onnx.graph.initializer
    }
    with torch.no_grad():
        model.policy.mlp_extractor.policy_net[0].weight.data.copy_(
            w["network_body._body_endoder.seq_layers.0.weight"]
        )
        model.policy.mlp_extractor.policy_net[0].bias.data.copy_(
            w["network_body._body_endoder.seq_layers.0.bias"]
        )
        model.policy.mlp_extractor.policy_net[2].weight.data.copy_(
            w["network_body._body_endoder.seq_layers.2.weight"]
        )
        model.policy.mlp_extractor.policy_net[2].bias.data.copy_(
            w["network_body._body_endoder.seq_layers.2.bias"]
        )
        model.policy.mlp_extractor.policy_net[4].weight.data.copy_(
            w["network_body._body_endoder.seq_layers.4.weight"]
        )
        model.policy.mlp_extractor.policy_net[4].bias.data.copy_(
            w["network_body._body_endoder.seq_layers.4.bias"]
        )
        model.policy.action_net.weight.data.copy_(
            w["action_model._continuous_distribution.mu.weight"]
        )
        model.policy.action_net.bias.data.copy_(
            w["action_model._continuous_distribution.mu.bias"]
        )
        model.policy.log_std.data.fill_(-3.5)

    # Freeze policy weights so value network aligns while keeping policy stable
    for p in model.policy.mlp_extractor.policy_net.parameters():
        p.requires_grad = False
    model.policy.action_net.weight.requires_grad = False
    model.policy.action_net.bias.requires_grad = False
    model.policy.log_std.requires_grad = False


def evaluate_policy(model: PPO, env: UnityToGymWrapper, num_episodes: int = 50) -> float:
    """Evaluates the model over num_episodes and prints the success rate."""
    print(f"\n--- Evaluating Model over {num_episodes} episodes ---")
    successes = 0
    for ep in range(num_episodes):
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        if reward > 2.0:
            successes += 1
    rate = successes / num_episodes
    print(f"Evaluation Result: {successes}/{num_episodes} reached goal ({rate * 100:.1f}%)")
    return rate


def main():
    binary_path = "/home/noah/Downloads/export/unitybuild.x86_64"
    onnx_path = os.path.join(os.path.dirname(__file__), "Easy.onnx")

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=20.0)

    unity_env = UnityEnvironment(
        file_name=binary_path,
        no_graphics=True,
        side_channels=[engine_channel],
    )
    env = UnityToGymWrapper(unity_env)

    policy_kwargs = dict(
        net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024]),
        activation_fn=nn.SiLU,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-5,
        n_steps=512,
        batch_size=64,
        device=device,
        verbose=0,
    )

    if os.path.exists(onnx_path):
        print(f"Loading pretrained weights from {onnx_path}...")
        load_pretrained_policy(model, onnx_path)

    stop_callback = SuccessRateStopCallback(target_success_rate=0.75, window_size=30)
    print("Starting training until 75% success rate is reached...")
    model.learn(total_timesteps=50000, callback=stop_callback)

    final_rate = evaluate_policy(model, env, num_episodes=50)
    print(f"Final Success Rate: {final_rate * 100:.1f}%")

    print("Saving model to unity_model.zip")
    model.save("unity_model")

    env.close()


if __name__ == "__main__":
    main()
