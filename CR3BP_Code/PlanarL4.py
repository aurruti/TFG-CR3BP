# Plotting key Lyapunov orbits around L4
# Aitor Urruticoechea 2022
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from aux_functions import *

# Basic operational data
mu_earthsun = load_basic_data()

# Data to be Imported
L1, L2, L4, L5 = import_LP()

# Linealized system gives the initial conditions
x = L4
y = np.sqrt(3)/2 # L4 location
A = np.array(CR3BP_A(x,y,mu_earthsun)) # Function from aux_functions

vaps = linalg.eigvals(A)
veps = linalg.eig(A)[1]
vaps = clear_small(vaps,10**(-10))
veps = clear_small(veps,10**(-10))

omega_vaps = np.imag(vaps)
C = remove_i(veps)
invC = linalg.inv(C)

check = clear_small(invC@A@C,10**(-10))

# Plots
#system_plot(mu_earthsun,L1,L2,L4,L5,xlim=[0.4998,0.50015],ylim=[0.8659, 0.8661],zoom=3,showSun=False, showEarth=False,showL1=False,showL2=False,showL5=False)
#system_plot(mu_earthsun,L1,L2,L4,L5,xlim=[0.472,0.527],ylim=[0.85, 0.882],zoom=1,showSun=False, showEarth=False,showL1=False,showL2=False,showL5=False)


# Objective function
def objective(ds0,s0,mu, style='none', step=5):
    s_fin, t = shoot_to_poincare_x(mu, [s0[0],s0[1],ds0[0],ds0[1]], style=style, bstep=step)
    s_f = []
    s_f.append(s_fin[1]+s_fin[2])
    s_f.append(s_fin[1]+s_fin[3])
    return np.array(s_f)-np.array([s0[1]+ds0[0], s0[1]+ds0[1]])

print("") 
print(">> L4 Planar Orbits <<")
print("")

fid1 = open('data/L4_PlanarOrbits.txt','w')
success_1 = 0
fail_1 = 0
success_2 = 0
fail_2 = 0
total = 0

tol = 10**(-8)
print('Long-period orbits')
can_continue = False
y0 = y
while not can_continue:
    y0 = y0 + 10**(-5)
    [t0_1, x0_1, dx_1, dy_1] = lin_eq_1(C, omega_vaps, 10**(-5), 10**(-5), y0)
    s0 = [float(x+x0_1),y0]
    root1 = optimize.root(objective, [dx_1,dy_1], args=(s0,mu_earthsun,'none',5), method='hybr', tol=10**(-10))
    s_fin, t_1 = shoot_to_poincare_x(mu_earthsun, [s0[0],s0[1],root1.x[0],root1.x[1]])
    if root1.success and (np.array([s0[0],s0[1],root1.x[0],root1.x[1]])-np.array(s_fin)<tol).all():
        #The initial orbit does incorporate some short-term movement so it is not plotted
        #print('Orbit at: [s0]; ds0; time')
        #print(s0)
        #print(str(root1.x[0]) + ' '+ str(root1.x[1]))
        #print(t_1)
        #shoot(mu_earthsun,[s0[0],s0[1],root1.x[0],root1.x[1]], t_1, style='-',res=3000)
        #fid1.write(str(s0[0]) + " " + str(s0[1]) + " " + str(root1.x[0]) + " " + str(root1.x[1]) + " " + str(t_1) + '\n')
        print('')
        can_continue = True

while y0 <= y + 10*10**(-5):
    y0 = y0 + 10**(-5)
    s0[1] = y0
    ds0 = [root1.x[0], root1.x[1]]
    root1 = optimize.root(objective, ds0, args=(s0,mu_earthsun,'none',5), method='hybr', tol=10**(-10))
    s_fin, t_1 = shoot_to_poincare_x(mu_earthsun, [s0[0],s0[1],root1.x[0],root1.x[1]])
    if root1.success and (np.array([s0[0],s0[1],root1.x[0],root1.x[1]])-np.array(s_fin)<tol).all():
        print('Orbit at: [s0]; ds0; time')
        print(s0)
        print(str(root1.x[0]) + ' '+ str(root1.x[1]))
        print(t_1)
        shoot(mu_earthsun,[s0[0],s0[1],root1.x[0],root1.x[1]], t_1, style='-',res=3000)
        fid1.write(str(s0[0]) + " " + str(s0[1]) + " " + str(root1.x[0]) + " " + str(root1.x[1]) + " " + str(t_1) + '\n')
        print('')

fid1.close()
legend_object1 = Patch(facecolor='C0', edgecolor='black')
legend_object2 = Patch(facecolor='C1', edgecolor='black')
legend_object3 = Patch(facecolor='C2', edgecolor='black')
legend_object4 = Patch(facecolor='C3', edgecolor='black')
legend_object5 = Patch(facecolor='C4', edgecolor='black')
legend_object6 = Patch(facecolor='C5', edgecolor='black')
legend_object7 = Patch(facecolor='C6', edgecolor='black')
legend_object8 = Patch(facecolor='C7', edgecolor='black')
legend_object9 = Patch(facecolor='C8', edgecolor='black')
legend_object10 = Patch(facecolor='C9', edgecolor='black')

plt.legend(handles=[legend_object1, legend_object2, legend_object3, legend_object4, legend_object5, legend_object6,legend_object7, legend_object8,legend_object9,legend_object10],labels=['','','','','','','','','', 'Long-term planar orbits'],ncols=11, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='upper left')
plt.show()
