from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from uav_comp import UavTrajectory

# deleted on 07.31 for Version 4 - 2
#env = UavTrajectory()

# added on 07.31 for Version 4 - 2
env = make_vec_env(UavTrajectory, n_envs=1)

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log = '/home/baseline/output/comp')

# edited on 08.15 for Version 8 - 1
model.learn(total_timesteps=200000, log_interval=1)

model.save('/home/baseline/output/comp/model') 

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
