from calc_cont_one_obj import CalculatePoint, CalculateAngle, LineCircle, AngleAngle
import math

pi = math.pi

class ContributionOfOneObject: 

    def __init__(self, x_obj, r_obj, WD, x_UAV, y_UAV, alpha1, alpha2, nxt):
        self.x_obj = x_obj
        self.r_obj = r_obj
        self.WD = WD
        self.x_UAV = x_UAV
        self.y_UAV = y_UAV
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.left_flag = nxt[0]
        self.right_flag = nxt[1]

    def _seperate_contribution(self):
        # FOV
        UAV_obj = CalculatePoint(self.x_UAV, self.y_UAV, self.x_obj, 0)
        beta = UAV_obj.calc_angle()
        contribution = LineCircle(self.x_UAV, self.y_UAV, self.WD, self.x_obj, self.r_obj)
        angle_FOV = contribution.cont_FOV(self.alpha1, self.alpha2, beta)

        # Blocking
        UAV_obj = CalculatePoint(self.x_obj, 0, self.x_UAV, self.y_UAV)
        alpha_center = UAV_obj.calc_angle()
        d = UAV_obj.calc_distance()
        d_alpha = math.acos(self.r_obj / d)
        Blocking = CalculateAngle(alpha_center, d_alpha)
        angle_B = (Blocking.minus(), Blocking.add())
        # renew
        UAV_block = UAV_obj.calc_angle_rev()
        d_block = math.asin(self.r_obj / d)
        UAV_B = CalculateAngle(UAV_block, d_block)
        angle_old = (UAV_B.minus(), UAV_B.add())

        # WD
        angle_WD = contribution.cont_WD()

        return angle_FOV, angle_B, angle_WD, angle_old 

    def total_contribution(self, idx, q, num):
        if math.sqrt(((self.x_obj-self.x_UAV)**2) + (self.y_UAV**2)) <= self.r_obj:
            return -1
        else:
            angle_FOV, angle_B, angle_WD, angle_old = self._seperate_contribution()

            # renew
            new = AngleAngle((self.alpha1, self.alpha2), angle_old)
            c = new.minus()

            if idx > 0 and self.left_flag:
                q.put((idx-1, c[0][0], c[0][1], (1,0)))
            if idx < num - 1 and self.right_flag:
                if len(c) == 2:
                    q.put((idx+1, c[1][0], c[1][1], (0,1)))
                else:
                    q.put((idx+1, c[0][0], c[0][1], (0,1)))
            print("Analysing NO.", idx, "...")

            # calculate contribution
            FOV_B = AngleAngle(angle_FOV, angle_B)
            dir_contribution = FOV_B.intervalIntersection()
            Dir_WD = AngleAngle(dir_contribution, angle_WD)
            contribution = Dir_WD.intervalIntersection()
            cont_up = AngleAngle(contribution, (0, pi))
            upper_contribution = cont_up.intervalIntersection() # only upper circle
            if upper_contribution == (0,0):
                return 0
            else:
                return upper_contribution 
