import math
from cont_one_obj import ContributionOfOneObject
from gen_obj import GenerateObject
import random
from calc_cont_one_obj import CalculateAngle
import queue

pi = math.pi

obj_num = 10
r_min = 15
r_max = 50
d_min = 10
d_max = 50
alpha_UAV = 42 / 180 * pi
WD = 6 * 36.5 * 2 # 438

objects = GenerateObject(obj_num, r_min, r_max, d_min, d_max)
r_obj_list, x_obj_list = objects.generate()

x_min = -10
x_max = 1000
y_min = 0
y_max = 1000 # actually it should be 6000

# x_UAV = -r_obj_list[0] - 10
# y_UAV = 0
x_UAV = random.uniform(x_min, x_max)
y_UAV = random.uniform(y_min, y_max)
alpha = random.uniform(7/6*pi, 11/6*pi)

first_obj = CalculateAngle(alpha, alpha_UAV)
alpha1 = first_obj.minus()
alpha2 = first_obj.add()

cont = [None] * obj_num
obj_q = queue.PriorityQueue()
for i in range(obj_num-1):
    if x_UAV <= x_obj_list[i+1] - r_obj_list[i+1]:
        obj_q.put((i, alpha1, alpha2, (1,1)))
        break
if obj_q.empty():
    obj_q.put((obj_num-1, alpha1, alpha2, (1,0)))

while not obj_q.empty():
    current_obj = obj_q.get()
    current_eva = ContributionOfOneObject(x_obj_list[current_obj[0]], r_obj_list[current_obj[0]], WD, x_UAV, y_UAV, current_obj[1], current_obj[2], current_obj[3])
    cont[current_obj[0]] = current_eva.total_contribution(current_obj[0], obj_q, obj_num)

print("radius: ", r_obj_list)
print("x of objects: ", x_obj_list)
print("x of UAV: ", x_UAV)
print("y of UAV: ", y_UAV)
print("alpha of UAV: ", alpha)
print("contribution: ", cont)
