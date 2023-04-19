import math
from sympy import *
from sympy.geometry import *

pi = math.pi

class CalculatePoint:

    def __init__(self, p1_x, p1_y, p2_x, p2_y):
        self.p1 = (p1_x, p1_y) # origin
        self.p2 = (p2_x, p2_y) # destination

    def calc_angle(self):
        # Difference in x coordinates
        dx = self.p2[0] - self.p1[0]

        # Difference in y coordinates
        dy = self.p2[1] - self.p1[1]

        # Angle between p1 and p2 in radians
        theta = math.atan2(dy, dx)
        if theta < 0:
            theta = theta + 2 * pi # theta belongs to [0,2pi)
        return (theta)

    def calc_angle_rev(self):
        # Difference in x coordinates
        dx = self.p1[0] - self.p2[0]

        # Difference in y coordinates
        dy = self.p1[1] - self.p2[1]

        # Angle between p1 and p2 in radians
        theta = math.atan2(dy, dx)
        if theta < 0:
            theta = theta + 2 * pi # theta belongs to [0,2pi)
        return (theta)

    def calc_distance(self):
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        d = math.sqrt(dx * dx + dy * dy)
        return (d)

class CalculateAngle:

    def __init__(self, theta, d_theta):
        self.theta = theta
        self.d_theta = d_theta

    def add(self):
        theta1 = self.theta + self.d_theta
        if theta1 >= 2 * pi:
            theta1 = theta1 - 2 * pi # theta1 belongs to [0,2pi)
        return (theta1)

    def minus(self):
        theta2 = self.theta - self.d_theta
        if theta2 < 0:
            theta2 = theta2 + 2 * pi # theta2 belongs to [0,2pi)
        return (theta2)

class LineCircle:

    def __init__(self, x_UAV, y_UAV, WD, x_obj, r_obj):
        self.UAV = Circle(Point(x_UAV, y_UAV), WD)
        self.object = Circle(Point(x_obj, 0), r_obj)
        self.WD = WD

    def _intersect(self, s1, s2, judge):
        p_list = intersection(s1, s2)
        if len(p_list) == 2:
            p1 = p_list[0].evalf()
            p2 = p_list[1].evalf()
            if judge == true:
                d1 = math.sqrt(((p1.x-self.UAV.center.x)**2) + ((p1.y-self.UAV.center.y)**2))
                d2 = math.sqrt(((p2.x-self.UAV.center.x)**2) + ((p2.y-self.UAV.center.y)**2))
                if d1 < d2:
                    return (p1)
                else:
                    return (p2)
            else:
                return (p1, p2)
        elif len(p_list) == 1:
            p = p_list.evalf()
            return (p)
    
    def cont_FOV(self, alpha_min, alpha_max, beta):
        p1 = Point(self.UAV.center.x + math.cos(alpha_min), self.UAV.center.y + math.sin(alpha_min))
        l1 = Line(self.UAV.center, p1)
        p2 = Point(self.UAV.center.x + math.cos(alpha_max), self.UAV.center.y + math.sin(alpha_max))
        l2 = Line(self.UAV.center, p2)

        p_left = self._intersect(l1, self.object, true)
        p_right = self._intersect(l2, self.object, true)
                    
        if p_left and p_right:
            left = CalculatePoint(self.object.center.x, self.object.center.y, p_left.x, p_left.y)
            angle_right = left.calc_angle() # different clockwise 
            right = CalculatePoint(self.object.center.x, self.object.center.y, p_right.x, p_right.y)
            angle_left = right.calc_angle()
            return (angle_left, angle_right)
        else:
            if alpha_max > alpha_min:
                if beta > alpha_min and beta < alpha_max:
                    return (0, pi)
            else:
                if beta > alpha_min or beta < alpha_max:
                    return (0, pi)
            return (0, 0) 

    def cont_WD(self):
        d = math.sqrt(((self.object.center.x-self.UAV.center.x)**2) + ((self.object.center.y-self.UAV.center.y)**2))
        if self.WD <= d - self.object.radius:
            return (0, 0)
        elif self.WD >= d + self.object.radius:
            return (0, 2 * pi)
        else:
            p1, p2 = self._intersect(self.UAV, self.object, false)
            left = CalculatePoint(self.object.center.x, self.object.center.y, p1.x, p1.y)
            angle1 = left.calc_angle() # different clockwise 
            right = CalculatePoint(self.object.center.x, self.object.center.y, p2.x, p2.y)
            angle2 = right.calc_angle()

            # to determine which is left
            p0 = self._intersect(Line(self.UAV.center, self.object.center), self.object, true) 
            middle = CalculatePoint(self.object.center.x, self.object.center.y, p0.x, p0.y)
            angle_mid = middle.calc_angle()
            if abs(angle1 + angle2 - 2*angle_mid) < abs(angle1 + angle2 - 2*angle_mid - pi) \
                and abs(angle1 + angle2 - 2*angle_mid) < abs(angle1 + angle2 - 2*angle_mid + pi):
                if angle1 < angle2:
                    return (angle1, angle2)
                else:
                    return (angle2, angle1)
            else:
                if angle1 < angle2:
                    return (angle2, angle1)
                else:
                    return (angle1, angle2)

class AngleAngle:

    def __init__(self, angle_tuple1, angle_tuple2):
        angle_list1 = list(angle_tuple1)
        angle_list2 = list(angle_tuple2)
        if angle_list1[1] < angle_list1[0]:
            angle_list1[1] = angle_list1[1] + 2*pi
        if angle_list2[1] < angle_list2[0]:
            angle_list2[1] = angle_list2[1] + 2*pi 

        self.a1, self.a2 = angle_list1
        self.b1, self.b2 = angle_list2

    def intervalIntersection(self):
        if self.b2 >= self.a1 and self.a2 >= self.b1:
            contribution_list = [max(self.a1, self.b1), min(self.a2, self.b2)]
            if contribution_list[1] >= 2*pi:
                contribution_list[1] = contribution_list[1] - 2*pi
            contribution = tuple(contribution_list)
            return contribution
        else:
            return (0,0)

    def minus(self):
        contribution_list2 = None
        if self.a2 >= self.b2 and self.b1 >= self.a1:
            contribution_list = [self.a1, self.b1]
            contribution_list2 = [self.b2, self.a2]
        elif self.b2 >= self.a2 and self.a2 >= self.b1 and self.b1 >= self.a1:
            contribution_list = [self.a1, self.b1]
        elif self.a2 >= self.b2 and self.b2 >= self.a1 and self.a1 >= self.b1:
            contribution_list = [self.b2, self.a2]
        else:
            contribution_list = [self.a1, self.a2]
        cont = []
        cont.append(self._rt(contribution_list))
        if contribution_list2:
            cont.append(self._rt(contribution_list2))
        print(cont)
        return cont

    def _rt(self, contribution_list):
        if contribution_list[1] >= 2*pi:
            contribution_list[1] = contribution_list[1] - 2*pi
        contribution = tuple(contribution_list)
        return contribution