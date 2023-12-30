import random
import math

pi = math.pi

alpha_left = 7/6*pi
alpha_right = 11/6*pi
speed_left = 0
speed_right = 16
dt = 1
distance_choice = [15]
angle_choice = [7/6*pi, 8/6*pi, 9/6*pi, 10/6*pi, 11/6*pi]

class Action:

    def __init__(self):
        pass

    def decode(self, idx):
        if idx < len(distance_choice):
            return (0, distance_choice[idx])
        elif idx < 2*len(distance_choice):
            return (0, -distance_choice[idx-len(distance_choice)])
        else:
            return (1, angle_choice[idx-2*len(distance_choice)])

    def getActionNum(self):
        return 2*len(distance_choice)+len(angle_choice)

    def random_action(self):
        return random.randint(0, self.getActionNum()-1)
