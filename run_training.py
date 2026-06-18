"""
Echos PPO — Training Script
============================
Auto-detects GPU (CUDA / ROCm / CPU).
"""
import platform
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from train import CurriculumEchosEnv, CurriculumCallback, make_env

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
is_linux = platform.system() == "Linux"

print(f"  Device: {device}  GPU: {gpu_name}  OS: {platform.system()}")

# Hyperparameters
if "MI300" in gpu_name:
    N_ENVS, N_STEPS, BATCH, NET_ARCH = 64, 16384, 8192, [2048, 2048]
elif "T4" in gpu_name:
    N_ENVS, N_STEPS, BATCH, NET_ARCH = 32, 8192, 4096, [1024, 1024]
else:
    N_ENVS, N_STEPS, BATCH, NET_ARCH = 8, 2048, 256, [256, 256]

TOTAL_STEPS = 1_000_000
LR = 3e-4

if __name__ == '__main__':
    print("=" * 60)
    print("  Echos PPO — Curriculum Training")
    print("=" * 60)
    print(f"  Envs: {N_ENVS}  Steps: {TOTAL_STEPS:,}  Device: {device}")
    print(f"  n_steps: {N_STEPS}  batch: {BATCH}  net: {NET_ARCH}")

    VecEnv = SubprocVecEnv if is_linux else DummyVecEnv
    vec_env = VecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", vec_env,
        verbose=1, learning_rate=LR,
        n_steps=N_STEPS, batch_size=BATCH, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=NET_ARCH),
        tensorboard_log="./echos_tb/",
        device=device,
    )

    cb = CurriculumCallback(verbose=1)

    print("\nStarting training...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb,
                progress_bar=True, reset_num_timesteps=True)

    model.save("echos_ppo")
    vec_env.save("echos_vecnorm.pkl")
    print("\nSaved: echos_ppo.zip, echos_vecnorm.pkl")

    eval_env = Monitor(CurriculumEchosEnv())
    rewards, lengths = evaluate_policy(model, eval_env, n_eval_episodes=10,
                                       deterministic=True, return_episode_rewards=True)
    print(f"\nEvaluation (10 episodes):")
    print(f"  Mean reward: {np.mean(rewards):.1f}  Max: {np.max(rewards):.1f}")
    print(f"  Mean length: {np.mean(lengths):.0f}")
    vec_env.close()
