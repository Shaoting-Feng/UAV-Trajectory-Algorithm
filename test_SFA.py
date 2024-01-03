from uav_random import UavTrajectory

uta = UavTrajectory()
for i in range(1, 11):
    uta.reset()
    terminated = False
    truncated = False
    obj_idx = 0
    while not (terminated or truncated):
        if uta.x_UAV < uta.x_obj_list[obj_idx]: 
            act_idx = 0
        else:
            act_idx = 4
            obj_idx = obj_idx + 1
        _, _, terminated, truncated, _ = uta.step(act_idx)
        if obj_idx >= 10:
            truncated = True # a new abnormal ending method