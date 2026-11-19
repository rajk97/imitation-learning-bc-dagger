"""Render one episode each of the expert and the trained BC student to mp4 (videos/)."""
import glob
import os

import gym
import imageio
import numpy as np
import torch

from rob831.infrastructure import pytorch_util as ptu
from rob831.infrastructure.utils import sample_trajectory
from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from rob831.policies.MLP_policy import MLPPolicySL

HW_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(HW_DIR, "videos")
MAX_STEPS = 1000


def load_student(env_name: str, env: gym.Env) -> MLPPolicySL:
    short = env_name.split("-")[0].lower()
    runs = sorted(glob.glob(os.path.join(HW_DIR, "data", f"q1_video_bc_{short}_{env_name}_*")), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError(f"No saved BC run for {env_name}; train with --save_params first")
    policy = MLPPolicySL(env.action_space.shape[0], env.observation_space.shape[0], n_layers=2, size=64)
    policy.load_state_dict(torch.load(os.path.join(runs[-1], "policy_itr_0.pt"), map_location="cpu"))
    return policy


def record(env: gym.Env, policy, out_path: str) -> None:
    path = sample_trajectory(env, policy, MAX_STEPS, render=True, render_mode=("rgb_array",))
    fps = int(round(1.0 / env.dt))
    imageio.mimsave(out_path, list(path["image_obs"]), fps=fps)
    print(f"{os.path.basename(out_path)}: {len(path['reward'])} steps, return {path['reward'].sum():.0f}")


def main() -> None:
    ptu.init_gpu(use_gpu=False)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    np.random.seed(0)
    for env_name, expert_file in [("Ant-v2", "Ant.pkl"), ("Humanoid-v2", "Humanoid.pkl")]:
        env = gym.make(env_name)
        env.reset(seed=0)
        expert = LoadedGaussianPolicy(os.path.join(HW_DIR, "rob831", "policies", "experts", expert_file))
        record(env, expert, os.path.join(VIDEO_DIR, f"{env_name}_expert.mp4"))
        record(env, load_student(env_name, env), os.path.join(VIDEO_DIR, f"{env_name}_bc.mp4"))


if __name__ == "__main__":
    main()
