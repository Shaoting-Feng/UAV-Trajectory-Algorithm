from uav_greedy import UavTrajectory

uta = UavTrajectory()
for i in range(1, 11):
    uta.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        r = float('-inf')
        for i in range(uta.a.getActionNum()):
            if uta.tiptoe(i) > r:
                act_idx = i
                r = uta.tiptoe(i)
        _, _, terminated, truncated, _ = uta.step(act_idx)