from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from uav import UavTrajectory

# deleted on 07.31 for Version 4 - 2
#env = UavTrajectory()

# added on 07.31 for Version 4 - 2
env = make_vec_env(UavTrajectory, n_envs=1)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = '/home/baseline/output')

# edited on 08.15 for Version 8 - 1
# model.learn(total_timesteps=100000, log_interval=1)

# edited on 10.08 for modifying base codes
model.learn(total_timesteps=10, log_interval=1)

model.save('/home/baseline/model') 

# deleted on 08.02 for Version 5 - 1
'''
# added on 07.31 for Version 4 - 2
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("console")
    if dones:
        break
'''
