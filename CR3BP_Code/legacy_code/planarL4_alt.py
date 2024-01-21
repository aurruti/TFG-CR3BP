# Plotting key Lyapunov orbits around L4
# Aitor Urruticoechea 2022
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
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
system_plot(mu_earthsun,L1,L2,L4,L5)
#shoot_to_poincare_x(mu_earthsun, [x+x0_2, y, dx_2, dy_2],style='-g',bstep=2)
#shoot_to_poincare_x(mu_earthsun, [x+x0_1, y, dx_1, dy_1],style='-b',bstep=5)

def objective(ds0,s0,mu, style='none', step=5):
    s_fin, t = shoot_to_poincare_x(mu, [s0[0],s0[1],ds0[0],ds0[1]], style=style, bstep=step)
    ds_fin = []
    #s_f.append(abs(s_fin[0])+abs(s_fin[2]))
    #s_f.append(abs(s_fin[1])+abs(s_fin[3]))
    #return np.array(s_f)-np.array(abs(np.array(s0))+abs(np.array(ds0)))
    ds_fin.append(s_fin[2])
    ds_fin.append(s_fin[3])
    ds_fin.append(s_fin[1])
    return np.array(ds_fin)-np.array([ds0[0], ds0[1], s0[1]])

print("_________________________________________________________________________________________________")
print("") 
print("                                     L4 Planar Orbits")
print("_________________________________________________________________________________________________")

fid1 = open('data/L4_PlanarOrbits_1.txt','w')
fid2 = open('data/L4_PlanarOrbits_2.txt','w')
success_1 = 0
fail_1 = 0
success_2 = 0
fail_2 = 0
total = 0
for y0 in range(1,100,5):
    y0 = y + y0*10**(-6)
    total = total+1
    [t0_1, x0_1, dx_1, dy_1] = lin_eq_1(C, omega_vaps, 10**(-5), 10**(-5), y0)
    [t0_2, x0_2, dx_2, dy_2] = lin_eq_2(C, omega_vaps, 10**(-5), 10**(-5), y0)
    root1 = optimize.root(objective, [dx_1,dy_1], args=([float(x+x0_1),y0],mu_earthsun,'none',5), method='hybr', tol=10**(-5))
    if root1.success: #Only record the solutions that actually converge
        fid1 = open('data/L4_PlanarOrbits_1.txt','a')
        success_1 = success_1 + 1
        print('Type-1 Orbit nº' + str(success_1) + ' at y0 = '+ str(y0) + ' | dx = ' + str(root1.x[0]) + ' | dy = ' + str(root1.x[1]) + '    (all in NDU)')
        fid1.write(str(y0) + " " + str(root1.x[0]) + " " + str(root1.x[1]) + '\n')
        fid1.close()
    else:
        [t0_1, x0_1, dx_1, dy_1] = lin_eq_1(C, omega_vaps, 10**(-4), 10**(-4), y0)
        root1 = optimize.root(objective, [dx_1,dy_1], args=([float(x+x0_1),y0],mu_earthsun,'none',5), method='hybr', tol=10**(-5))
        if root1.success: #Only record the solutions that actually converge
            fid1 = open('data/L4_PlanarOrbits_1.txt','a')
            success_1 = success_1 + 1
            print('Type-1 Orbit nº' + str(success_1) + ' at y0 = '+ str(y0) + ' | dx = ' + str(root1.x[0]) + ' | dy = ' + str(root1.x[1]) + '    (all in NDU)')
            fid1.write(str(y0) + " " + str(root1.x[0]) + " " + str(root1.x[1]) + '\n')
            fid1.close()
        else:
            fail_1 = fail_1 + 1

    # Same with type-2 orbits
    root2 = optimize.root(objective, [dx_2,dy_2], args=([float(x+x0_2),y0],mu_earthsun,'none',5), method='hybr', tol=10**(-5))
    if root2.success:
        fid2 = open('data/L4_PlanarOrbits_2.txt','a')
        success_2 = success_2 + 1
        print('Type-2 Orbit nº' + str(success_2) + ' at y0 = '+ str(y0) + ' | dx = ' + str(root2.x[0]) + ' | dy = ' + str(root2.x[1]) + '    (all in NDU)')
        fid2.write(str(y0) + " " + str(root2.x[0]) + " " + str(root2.x[1]) + '\n')
        fid2.close()
    else:
        [t0_2, x0_2, dx_2, dy_2] = lin_eq_2(C, omega_vaps, 10**(-4), 10**(-4), y0)
        root2 = optimize.root(objective, [dx_2,dy_2], args=([float(x+x0_2),y0],mu_earthsun,'none',5), method='hybr', tol=10**(-5))
        if root2.success: #Only record the solutions that actually converge
            fid2 = open('data/L4_PlanarOrbits_2.txt','a')
            success_2 = success_2 + 1
            print('Type-2 Orbit nº' + str(success_2) + ' at y0 = '+ str(y0) + ' | dx = ' + str(root2.x[0]) + ' | dy = ' + str(root2.x[1]) + '    (all in NDU)')
            fid2.write(str(y0) + " " + str(root2.x[0]) + " " + str(root2.x[1]) + '\n')
            fid2.close()
        else:
            fail_2 = fail_2 + 1

print('')
print('___________________________________________________________________________________')
print('                  |      Success Ratio      |      Failure to converge Ratio      |')
print('Type-1 orbits:    |         ' + str(round((success_1/total * 100),2)) + '%           |       ' + str(round((fail_1/total * 100),2)) + '%')            
print('Type-2 orbits:    |         ' + str(round((success_2/total * 100),2)) + '%           |       ' + str(round((fail_2/total * 100),2)) + '%')
print('___________________________________________________________________________________')
print('')
print('Note: plots correspond to the largest amplitude found for both types of orbit.')
shoot_to_poincare_x(mu_earthsun, [x+x0_1, y0, root1.x[0], root1.x[1]], style='-m', bstep=2)
shoot_to_poincare_x(mu_earthsun, [x+x0_1, y0, root2.x[0], root2.x[1]], style='-g', bstep=2)
plt.show()
