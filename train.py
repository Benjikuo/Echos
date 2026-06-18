"""
Echos — Curriculum Training Wrapper
====================================
Stage 0: Survivor close (2-4 m), no RTH required. Learn gradient climb.
Stage 1: Full maze, RTH required. Learn complete mission loop.
Stage 2: Randomized survivor position. Generalize.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sim import EchosEnv


class CurriculumEchosEnv(EchosEnv):
    def __init__(self):
        super().__init__()
        self.curriculum_stage = 0
        self._stage0_no_rth = False

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)

        if self.curriculum_stage == 0:
            angle = np.random.uniform(-np.pi / 3, np.pi / 3)
            dist = np.random.uniform(2.0, 4.0)
            self.survivor_pos = np.array([
                self.uav_pos[0] + dist * np.cos(angle),
                self.uav_pos[1] + dist * np.sin(angle),
            ])
            self.survivor_pos[0] = np.clip(self.survivor_pos[0], 0.5, 11.5)
            self.survivor_pos[1] = np.clip(self.survivor_pos[1], 0.5, 8.5)
            self._stage0_no_rth = True
        else:
            self._stage0_no_rth = False

        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._stage0_no_rth and self.commander.state == "RTH":
            terminated = True
            reward += 15.0
        info["curriculum_stage"] = self.curriculum_stage
        return obs, reward, terminated, truncated, info


class CurriculumCallback(BaseCallback):
    STAGE_THRESHOLDS = {0: 10.0, 1: 12.0}
    WINDOW = 50

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_stage = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        if len(self.episode_rewards) >= self.WINDOW:
            mean_r = np.mean(self.episode_rewards[-self.WINDOW:])
            if self.current_stage in self.STAGE_THRESHOLDS:
                thresh = self.STAGE_THRESHOLDS[self.current_stage]
                if mean_r >= thresh:
                    self.current_stage += 1
                    try:
                        self.training_env.set_attr("curriculum_stage", self.current_stage)
                    except Exception:
                        pass
                    if self.verbose:
                        print(f"\n[Curriculum] mean={mean_r:.1f} >= {thresh} -> Stage {self.current_stage}\n")
        return True


def make_env(rank, seed=0):
    def _init():
        env = CurriculumEchosEnv()
        env.reset(seed=seed + rank)
        return env
    return _init
