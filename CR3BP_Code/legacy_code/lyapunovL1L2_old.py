# Plotting key Lyapunov orbits around L1 and L2
# data form JPL-NASA - Park, R.S., et al., 2021
# Aitor Urruticoechea 2022
import numpy as np
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt
from aux_functions import *

# Basic operational data
mu_earthsun = load_basic_data()

# Data to be Imported
L1, L2, L4, L5 = import_LP()
Initial_Conditions_L1 = np.loadtxt('data/JPLdata_L1.txt')
Initial_Conditions_L2 = np.loadtxt('data/JPLdata_L2.txt')

# Function to find the solution of
def objective(variable,x0):
    vy0 = variable[0]
    tspan = variable[1]
    s = [x0, 0, 0, vy0]
    sol = shoot(mu_earthsun, s, tspan,'none')
    try:
        a = sol.y[0,99] # req.1: x_f = x_0
        b = sol.y[1,99] # req.2: y_f = 0
        # c = sol.y[2,99] # req.3: vx_f = 0
    except:
        print(sol)
    return a-x0, b

system_plot(mu_earthsun,L1,L2,L4,L5,[0.97, 1.02],[-0.025, 0.025])

print("_________________________________________________________________________________________________")
print("") 
print("                                     L1 Lyapunov Orbits")
print("_________________________________________________________________________________________________")

fid = open('data/L1_LyapunovOrbits.txt','w')
for i in range(np.size(Initial_Conditions_L1,0)):
    x0 = Initial_Conditions_L1[i,0]
    vy0_guess = Initial_Conditions_L1[i,1]
    tspan_guess = Initial_Conditions_L1[i,2]
    root = optimize.root(objective, [vy0_guess, tspan_guess], args=x0, method='hybr',tol=10**(-10))
    print("(" + str(i) + ") | x0 = " + str(x0) + " | vy0 = " + str(root.x[0]) + " | tspan = " + str(2*root.x[1]) + "        (all in NDU)")
    shoot(mu_earthsun,[x0,0,0,root.x[0]],root.x[1],'-')
    fid.write(str(x0) + " " + str(root.x[0]) + " " + str(root.x[1]) + '\n')
    
print("")
print("_________________________________________________________________________________________________")
print("")
print("                                     L2 Lyapunov Orbits")
print("_________________________________________________________________________________________________")

fid = open('data/L2_LyapunovOrbits.txt','w')
for i in range(np.size(Initial_Conditions_L2,0)):
    x0 = Initial_Conditions_L2[i,0]
    vy0_guess = Initial_Conditions_L2[i,1]
    tspan_guess = Initial_Conditions_L2[i,2]
    root = optimize.root(objective, [vy0_guess, tspan_guess], args=x0, method='hybr',tol=10**(-10))
    print("(" + str(i) + ") | x0 = " + str(x0) + " | vy0 = " + str(root.x[0]) + " | tspan = " + str(2*root.x[1]) + "        (all in NDU)")
    shoot(mu_earthsun,[x0,0,0,root.x[0]],root.x[1],'-')
    fid.write(str(x0) + " " + str(root.x[0]) + " " + str(root.x[1]) + '\n')

plt.show()


