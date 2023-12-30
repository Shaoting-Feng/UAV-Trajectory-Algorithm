from ..Game.action import dt

# Blade Profile Power
K = 570
v_b = 100

# Parasite Power
p = 1.225
F = 0.4

# Induced Power
m = 5
g = 9.8
A = 0.25

# Computing Energy
P_c = 20

def move_energy(distance):
    v = distance / dt

    # Blade Profile Power
    P_blade = K * (1 + 3*v**2/v_b**2)
    #print(P_blade)

    # Parasite Power
    P_parasite = 0.5 * p * v**3 * F
    #print(P_parasite)

    # Induced Power
    v_i = ((-v**2+(v**4+(m*g/p/A)**2)**0.5)/2)**0.5
    P_induced = m * g * v_i
    #print(P_induced)

    E = (P_blade+P_parasite+P_induced+P_c) * dt

    return E

def photo_energy():
    return move_energy(0)

'''
print(move_energy(0))
print(move_energy(15))
'''
