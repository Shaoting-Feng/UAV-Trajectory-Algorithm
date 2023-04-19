from sympy import *
from sympy.geometry import *
x, y = symbols('x y')
p1 = Point(0, 0)
p2 = Point(1, 1)
l1 = Line(p1, p2)
c1 = Circle(Point(0, 0), 1)
p = intersection(l1, c1)
print(p)