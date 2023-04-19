import random

class GenerateObject:

    def __init__(self, num, r_min, r_max, d_min, d_max):
        self.num = num
        self.r_min = r_min
        self.r_max = r_max
        self.d_min = d_min
        self.d_max = d_max

    def generate(self):
        r_obj_list = []

        # for test
        #r_obj_list.append(1)
        #for i in range(self.num-1):

        for i in range(self.num):
            r_obj_list.append(random.uniform(self.r_min, self.r_max)) 

        distance_x_obj_list = []
        for i in range(self.num-1):
            distance_x_obj_list.append(random.uniform(self.d_min, self.d_max)) 

        x_obj_list = []
        x_obj_list.append(0) # start with x = 0
        for i in range(self.num-1):
            x_obj_list.append(x_obj_list[i] + r_obj_list[i] + distance_x_obj_list[i] + r_obj_list[i+1])

        return r_obj_list, x_obj_list 

