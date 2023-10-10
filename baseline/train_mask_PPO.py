from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from uav import UavTrajectory
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
import numpy as np

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    #return env.valid_action_mask()
    action_mask = np.ones(7)
    action_mask[::2] = 0
    #1010101 --> 6622244220 --> 1的位的动作可以被选择
    #0101010 --> 33
    return action_mask

env = make_vec_env(UavTrajectory, n_envs=1)
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=2, verbose=1, tensorboard_log = '/home/baseline/output')
model.learn(5_000)

evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

model.save('/home/baseline/model')
del model # remove to demonstrate saving and loading

model = MaskablePPO.load('/home/baseline/model')

obs, _ = env.reset()
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
