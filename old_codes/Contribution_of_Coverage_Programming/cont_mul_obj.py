import math
from old_codes.Contribution_of_Coverage_Programming.cont_one_obj import ContributionOfOneObject
from old_codes.Contribution_of_Coverage_Programming.calc_cont_one_obj import CalculateAngle
import queue
import pickle

pi = math.pi

alpha_UAV = 42 / 180 * pi
WD = 6 * 36.5 * 2 # 438

class Reward:
    # edited on 08.15 for Version 8 - 3
    def __init__(self, x_UAV, y_UAV, alpha, r_obj_list, x_obj_list, obj_num):

        self.x_UAV = x_UAV
        self.y_UAV = y_UAV
        self.alpha = alpha

        # added on 08.15 for Version 8 -3
        self.r_obj_list = r_obj_list
        self.x_obj_list = x_obj_list
        self.obj_num = obj_num

    def get_reward(self):
        # deleted on 08.15 for Version 8 - 3
        '''
        with open('/home/baseline/output/objects.pkl', 'rb') as f:
            data = pickle.load(f)
        r_obj_list = data["r"]
        x_obj_list = data["x"]
        obj_num = data["num"] 
        '''

        first_obj = CalculateAngle(self.alpha, alpha_UAV)
        alpha1 = first_obj.minus()
        alpha2 = first_obj.add()

        cont = [None] * self.obj_num
        obj_q = queue.PriorityQueue()
        for i in range(self.obj_num-1):
            if self.x_UAV <= self.x_obj_list[i+1] - self.r_obj_list[i+1]:
                obj_q.put((i, alpha1, alpha2, (1,1)))
                break
        if obj_q.empty():
            obj_q.put((self.obj_num-1, alpha1, alpha2, (1,0)))

        while not obj_q.empty():
            current_obj = obj_q.get()
            current_eva = ContributionOfOneObject(self.x_obj_list[current_obj[0]], self.r_obj_list[current_obj[0]], WD, self.x_UAV, self.y_UAV, current_obj[1], current_obj[2], current_obj[3])
            cont[current_obj[0]] = current_eva.total_contribution(current_obj[0], obj_q, self.obj_num)

        return cont