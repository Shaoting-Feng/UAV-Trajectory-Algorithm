from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from uav import UavTrajectory

# deleted on 07.31 for Version 4 - 2
#env = UavTrajectory()

# added on 07.31 for Version 4 - 2
env = make_vec_env(UavTrajectory, n_envs=1)

model = PPO.load('/home/baseline/model')

# added on 07.31 for Version 4 - 2
obs = env.reset()

# added on 08.19 for Version 8.2
for i in range(10):

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("console")
        if dones:
            break