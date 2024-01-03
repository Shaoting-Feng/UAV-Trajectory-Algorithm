from uav_random import UavTrajectory
import random

uta = UavTrajectory()
for i in range(1, 11):
    uta.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        idx = random.randint(0, uta.a.getActionNum() - 1)
        _, _, terminated, truncated, _ = uta.step(idx)