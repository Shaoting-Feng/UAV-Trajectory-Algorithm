import numpy as np
from old_codes.Contribution_of_Coverage_Programming.gen_obj import GenerateObject
from old_codes.Game.action import Action
import math
pi = math.pi
from old_codes.Contribution_of_Coverage_Programming.energy import move_energy, photo_energy
from old_codes.Contribution_of_Coverage_Programming.cont_mul_obj import Reward

# edited on 08.15 for Version 8 - 1
obj_num = 10

r_min = 15
r_max = 50
d_min = 10
d_max = 50
segment = 10
y_UAV = 100 # fly at certain height
MOVE_AWAY_MINUS = 1
COLLIDE_MINUS = 1000
REPETE_PHOTO_MINUS = 1
done_factor = 0.8

# edited on 07.31 for Version 4 - 1
bandwidth = 0.1

# edited on 07.29 for Version 1 - 1
energy_importance = 0.1

# max_photo_num used for limit the steps in one episode within 8
# edited on 08.16 for Version 8 - 1
max_photo_num = 80
# set to 60 results in UAV reaches 465 / 880

# added on 07.29 for Version 1 - 4
action_batch = []
reward_batch = []

class UavTrajectory():
    def __init__(self):
        # added on 08.12 for Version 7 - 2
        objects = GenerateObject(obj_num, r_min, r_max, d_min, d_max)
        self.r_obj_list, self.x_obj_list = objects.generate()

        # Define action and observation space
        # They must be gym.spaces objects
        # Use discrete actions
        self.a = Action()

        # Initialize the agent
        self.x_UAV = self.x_obj_list[0] - self.r_obj_list[0] - 10
        
        # Initialize contribution record
        self.cont = [0] * obj_num
        self.period_cont = np.zeros((obj_num, segment))
        self.episode_r = 0

        # Initialize termination judgement
        # photo_num turns to denote number of actions
        self.photo_num = 0

        # Initialize observation
        self.last_act = -1

        # added on 08.01
        self.episode = 0

        # added on 10.18 to mask action on the top layer
        self.last_move_act = 0

        # added on 10.18 to record three aspects
        self.cov = 0
        self.ene = 0
        self.bdw = 0


    def getObs(self):
        # edited on 07.29 for Version 1 - 2
        obs = np.zeros((obj_num, segment+3))

        for i in range(obj_num):
            obs_1 = np.append(self.period_cont[i], self.x_UAV - self.x_obj_list[i])

            # added on 07.29 for Version 1 - 2
            obs_2 = np.append(obs_1, self.last_act)
            obs[i] = np.append(obs_2, self.photo_num)
        return obs


    def reset(self):
        # added on 08.12 for Version 7 - 2
        objects = GenerateObject(obj_num, r_min, r_max, d_min, d_max)
        self.r_obj_list, self.x_obj_list = objects.generate()

        # Initialize the agent at the right of the grid
        self.x_UAV = self.x_obj_list[0] - self.r_obj_list[0] - 10

        # Initialize contribution record
        self.cont = [0] * obj_num
        self.period_cont = np.zeros((obj_num, segment))
        self.episode_r = 0

        # Initialize termination judgement
        self.photo_num = 0

        # Initialize observation
        self.last_act = -1

        # added on 10.18 to mask action on the top layer
        self.last_move_act = 0

        # added on 10.18 to record three aspects
        self.cov = 0
        self.ene = 0
        self.bdw = 0

        return self.getObs().astype(np.float32), {}  # empty info dict
    

    def amplifyEnergy(self, E):
        # edited on 07.31 for Version 4 - 1
        return - (E - 1028.27) / (1703.00 - 1028.27) * energy_importance
    
        
    def renew_period(self, i, new_cont):
        for period in range(segment):
            if new_cont[0] < (period + 1) * pi/segment:
                if new_cont[1] < (period + 1) * pi/segment:
                    self.period_cont[i][period] = self.period_cont[i][period] + new_cont[1] - new_cont[0]
                else:
                    self.period_cont[i][period] = self.period_cont[i][period] + (period + 1) * pi/segment - new_cont[0]
                    for period2 in range(period+1, segment):
                        if new_cont[1] < (period2 + 1) * pi/segment:
                            self.period_cont[i][period2] = self.period_cont[i][period2] + new_cont[1] - period2 * pi/segment
                            break
                        else:
                            self.period_cont[i][period2] = self.period_cont[i][period2] + pi/segment
                break


    def renew_record(self, new_cont):
        new_r = [0] * obj_num
        for i in range(obj_num):
            if new_cont[i] != 0:
                if self.cont[i] == 0:
                    new_r[i] = new_cont[i][1] - new_cont[i][0]
                    self.cont[i] = [new_cont[i]]
                    self.renew_period(i, [new_cont[i][0], new_cont[i][1]])
                else:
                    tmp1 = len(self.cont[i]) # where the left of new_cont is 
                    tmp2 = len(self.cont[i]) # where the right of new_cont is
                    odd_flag1 = False
                    odd_flag2 = False
                    # find where the left of new_cont is 
                    for j in range(len(self.cont[i])):
                        if new_cont[i][0] <= self.cont[i][j][0]:
                            tmp1 = j
                            odd_flag1 = False # between two cont (tmp1-1 to tmp1)
                            break
                        elif new_cont[i][0] <= self.cont[i][j][1]:
                            tmp1 = j
                            odd_flag1 = True # in one cont (tmp1)
                            break
                    # find where the right of new_cont is
                    for j in range(tmp1, len(self.cont[i])):
                        if new_cont[i][1] <= self.cont[i][j][0]:
                            tmp2 = j
                            odd_flag2 = False
                            break
                        elif  new_cont[i][1] <= self.cont[i][j][1]:
                            tmp2 = j
                            odd_flag2 = True
                            break
                    if tmp1 == tmp2:
                        # cont[i][tmp1-1][1] < left < right < cont[i][tmp1][0]
                        if not odd_flag1 and not odd_flag2:
                            new_r[i] = new_cont[i][1] - new_cont[i][0]
                            self.cont[i].append(new_cont[i])
                            if tmp1 < len(self.cont[i])-1:
                                for j in range(tmp1+1, len(self.cont[i])):
                                    self.cont[i][j] = self.cont[i][j-1]
                                self.cont[i][tmp1] = new_cont[i]
                                self.renew_period(i, [new_cont[i][0], new_cont[i][1]])
                        # cont[i][tmp1][0] < left < right < cont[i][tmp1][1]
                        elif odd_flag1 and odd_flag2: 
                            new_r[i] = 0
                        # cont[i][tmp1-1][1] < left < cont[i][tmp1][0] < right < cont[i][tmp1][1]
                        elif not odd_flag1 and odd_flag2: 
                            new_r[i] = self.cont[i][tmp1][0] - new_cont[i][0]
                            self.renew_period(i, [new_cont[i][0], self.cont[i][tmp1][0]])
                            self.cont[i][tmp1][0] = new_cont[i][0]         
                    else:
                        # cont[i][tmp1-1][1] < left < cont[i][tmp1][0] < ... < cont[i][tmp2-1][1] < right < cont[i][tmp2][0]
                        if not odd_flag1 and not odd_flag2:
                            new_r[i] = self.cont[i][tmp1][0] - new_cont[i][0]
                            self.renew_period(i, [new_cont[i][0], self.cont[i][tmp1][0]])
                            new_r[i] = new_r[i] + new_cont[i][1] - self.cont[i][tmp2-1][1]
                            self.renew_period(i, [self.cont[i][tmp2-1][1], new_cont[i][1]])
                            for j in range(tmp1, tmp2-1):
                                new_r[i] = new_r[i] + self.cont[i][j+1][0] - self.cont[i][j][1]
                                self.renew_period(i, [self.cont[i][j][1], self.cont[i][j+1][0]])
                            self.cont[i][tmp1] = new_cont[i]
                            for j in range(tmp1+1, len(self.cont[i])-tmp2+tmp1+1):
                                self.cont[i][j] = self.cont[i][j+tmp2-tmp1-1]
                            for j in range(tmp2-tmp1-1):
                                self.cont[i].pop()
                        # cont[i][tmp1][0] < left < cont[i][tmp1][1] < ... < cont[i][tmp2][0] < right < cont[i][tmp2][1]
                        elif odd_flag1 and odd_flag2:
                            new_r[i] = 0
                            for j in range(tmp1, tmp2):
                                new_r[i] = new_r[i] + self.cont[i][j+1][0] - self.cont[i][j][1]
                                self.renew_period(i, [self.cont[i][j][1], self.cont[i][j+1][0]])
                            self.cont[i][tmp1] = [self.cont[i][tmp1][0], self.cont[i][tmp2][1]]
                            for j in range(tmp1+1, len(self.cont[i])-tmp2+tmp1):
                                self.cont[i][j] = self.cont[i][j+tmp2-tmp1]
                            for j in range(tmp2-tmp1):
                                self.cont[i].pop()
                        # cont[i][tmp1-1][1] < left < cont[i][tmp1][0] < ... < cont[i][tmp2][0] < right < cont[i][tmp2][1]
                        elif not odd_flag1 and odd_flag2: 
                            new_r[i] = self.cont[i][tmp1][0] - new_cont[i][0]
                            self.renew_period(i, [new_cont[i][0], self.cont[i][tmp1][0]])
                            for j in range(tmp1, tmp2):
                                new_r[i] = new_r[i] + self.cont[i][j+1][0] - self.cont[i][j][1]
                                self.renew_period(i, [self.cont[i][j][1], self.cont[i][j+1][0]])
                            self.cont[i][tmp1] = [new_cont[i][0], self.cont[i][tmp2][1]]
                            for j in range(tmp1+1, len(self.cont[i])-tmp2+tmp1):
                                self.cont[i][j] = self.cont[i][j+tmp2-tmp1]
                            for j in range(tmp2-tmp1):
                                self.cont[i].pop()
                        # cont[i][tmp1][0] < left < cont[i][tmp1][1] < ... < cont[i][tmp2-1][1] < right < cont[i][tmp2][0]
                        elif odd_flag1 and not odd_flag2: 
                            new_r[i] = new_r[i] + new_cont[i][1] - self.cont[i][tmp2-1][1]
                            self.renew_period(i, [self.cont[i][tmp2-1][1], new_cont[i][1]])
                            for j in range(tmp1, tmp2-1):
                                new_r[i] = new_r[i] + self.cont[i][j+1][0] - self.cont[i][j][1]
                                self.renew_period(i, [self.cont[i][j][1], self.cont[i][j+1][0]])
                            self.cont[i][tmp1] = [self.cont[i][tmp1][0], new_cont[i][1]]
                            for j in range(tmp1+1, len(self.cont[i])-tmp2+tmp1+1):
                                self.cont[i][j] = self.cont[i][j+tmp2-tmp1-1]
                            for j in range(tmp2-tmp1-1):
                                self.cont[i].pop()
        return new_r
    

    def check_photo(self):
        photo_sum = 0
        for i in range(obj_num):
            if self.cont[i] != 0:
                for j in range(len(self.cont[i])):
                    photo_sum = photo_sum + self.cont[i][j][1] - self.cont[i][j][0]
        if photo_sum > obj_num * pi * done_factor:
            done1 = True
            done2 = False
        elif self.photo_num >= max_photo_num:
            done1 = False
            done2 = True
        else:
            done1 = False
            done2 = False
        return done1, done2


    def step(self, action):
        concrete_action = self.a.decode(action)

        # added on 10.18 to mask actions on the top layer --> not completely random
        if self.last_act == action and concrete_action[0] == 1:
            action = self.last_move_act
            concrete_action = self.a.decode(action)

        # added on 07.29 for Version 1 - 4
        action_batch.append(action)

        terminated = False
        truncated = False
        if concrete_action[0] == 0:
            e = move_energy(abs(concrete_action[1]))
            # e is positive; bigger e means bigger energy consumption

            # edited on 07.29 for Version 1 - 1 
            r =  self.amplifyEnergy(e)

            # added on 10.18 to record three aspects
            self.ene = self.ene + e

            self.x_UAV = self.x_UAV + concrete_action[1]
            move_away_flag = True
            for i in range(obj_num):
                if math.sqrt(((self.x_obj_list[i]-self.x_UAV)**2) + (y_UAV**2)) <= self.r_obj_list[i]:
                    r = 0 - COLLIDE_MINUS
                    terminated = True
                if abs(self.x_obj_list[i]-self.x_UAV) < abs(self.x_obj_list[i]-self.x_UAV+concrete_action[1]):
                    move_away_flag = False
            if move_away_flag == True:
                r = r - MOVE_AWAY_MINUS
            self.episode_r = self.episode_r + r
            self.render_info = True

            # added on 07.29 for Version 1 - 3
            self.photo_num = self.photo_num + 1  
            terminated, truncated = self.check_photo()

            # added on 10.18 to handle last move action
            self.last_move_act = action
        else:
            # edited on 08.15 for Version 8 - 3
            R = Reward(self.x_UAV, y_UAV, concrete_action[1], self.r_obj_list, self.x_obj_list, obj_num)
            
            simp_r = R.get_reward()
            self.new_r = self.renew_record(simp_r)
            e = photo_energy()

            # edited on 07.29 for Version 1 - 1 
            r =  self.amplifyEnergy(e)

            # added on 07.31 for Version 4 - 1
            r = r - bandwidth

            self.photo_num = self.photo_num + 1  
            sumR = 0   

            # added on 07.29 for Version 1 - 1
            for i in range(obj_num):
                sumR = sumR + self.new_r[i]
            r = r + sumR / 4 # modified on 11.08

            terminated, truncated = self.check_photo()

            # added on 10.18 to record three aspects
            self.cov = self.cov + sumR
            self.bdw = self.bdw + 1
            
            self.render_info = False
        self.last_act = action

        # Optionally we can pass additional info, we are not using that for now
        info = {"contribution": self.cont}

        # added on 07.29 for Version 1 - 1
        if r > 1:
            r = 1
        elif r < -1:
            r = -1

        # added on 07.29 for Version 1 - 4
        reward_batch.append(r)

        # added on 07.29 for Version 1 - 4
        if terminated or truncated:
            # added on 08.01
            self.episode = self.episode + 1
            print("This is episode:", self.episode)

            # added on 08.15 for Version 7.2
            print("object_x:", self.x_obj_list)
            
            print("action_batch:", action_batch)
            action_batch.clear()
            print("reward_batch:", reward_batch)
            print("contribution of this episode:", self.cont)

            # added on 07.31 for Version 4 - 1
            total_reward = sum(reward_batch)
            print("reward of this episode:", total_reward)

            # added on 10.18 to record three aspects
            print("coverage of this episode:", self.cov)
            print("energy of this episode:", self.ene)
            print("bandwidth of this episode:", self.bdw)

            reward_batch.clear()

        return (
            self.getObs().astype(np.float32),
            r,
            terminated,
            truncated,
            info,
        )


    def render(self):
        pass

    def close(self):
        pass
