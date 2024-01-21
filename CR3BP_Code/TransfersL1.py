# Transfers from L1
# Aitor Urruticoechea 2022
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from aux_functions import *
from scipy import linalg

# Basic operational data
mu_earthsun = load_basic_data()

# Data to be Imported
L1, L2, L4, L5 = import_LP()

# Plots
#system_plot(mu_earthsun,L1,L2,L4,L5,xlim=[0.97,1.02],ylim=[-0.025, 0.025],zoom=1, showSun=False, showL4=False, showL5=False)
system_plot(mu_earthsun,L1,L2,L4,L5,xlim=[0.4,1.1],ylim=[-0.1,1],zoom=0,showSun=False,showL5=False)

legend_object1 = Patch(facecolor='C0', edgecolor='black')
legend_object2 = Patch(facecolor='C1', edgecolor='black')
legend_object3 = Patch(facecolor='C5', edgecolor='black')
#plt.legend(handles=[legend_object1, legend_object2, legend_object3],
#           labels=['Departure L1 Lyapunov Orbit', 'Trajectories within the L1-L4 Manifold', 'Optimized trajectory'],ncols=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='lower left')
plt.legend(handles=[legend_object2, legend_object3],
           labels=['Trajectories within the L1-L4 Manifold','Optimized trajectory'],ncols=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='lower left')



# Initial conditions: periodic orbit
L1_orbits = np.loadtxt('data/L1_LyapunovOrbits.txt')
L4_orbits = np.loadtxt('data/L4_PlanarOrbits.txt')
s0_L4 = np.delete(L4_orbits[-1,:],4,0) # largest
s0 = np.delete(L1_orbits[-1,:],4,0) # largest

# Propagation for an orbital period, gives the STM
s_end_orbit, t_orbit = shoot_to_poincare_y(mu_earthsun, s0, style='C0')
phi = variationals_prop(mu_earthsun, s0, t_orbit)

# Eigenvalues and eigenvectors of the STM gives the direction of instability
vaps = linalg.eigvals(phi)
#print(vaps) #(this is to check the values are the expected one - or close enough)
veps = linalg.eig(phi)[1]
found = False
n = -1
while not found:
    n = n+1
    if abs(vaps[n])>1.05: found = True
unstable_dir = veps[n,:] / np.linalg.norm(veps[n,:]) # Normalized unstable direction
if unstable_dir[0] > 0: unstable_dir = -unstable_dir
# Exploring that instability
maximum_y = [0,0,0,0]
s_transfer = [0,0,0,0]
h_transfer = 0
for i in range(1,500,5):
    h = i*10**(-7)
    s_new = s0 + h*unstable_dir
    s_end, t_end = shoot_to_poincare_x(mu_earthsun, s_new, style=None, crossings=1, x_obj=L4)
    shoot(mu_earthsun, s_new, t_end, style='C1',res=1000)
    if s_end[1] > maximum_y[1]:
        maximum_y = s_end
        s_transfer = s_new
        h_transfer = h

shoot_to_poincare_x(mu_earthsun, s_transfer, style='C5', bstep=0.1, crossings=1, x_obj=L4)
print("Manifold's closest approach to L4 at y: " + str(maximum_y[1]))
print("For an initial perturbation of h: " + str(h_transfer))

#plt.legend(handles=[legend_object2, legend_object3],
#           labels=['Trajectories within the L1-L4 Manifold','Optimized trajectory'],ncols=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='lower left')

s_end_orbit, t_orbit = shoot_to_poincare_y(mu_earthsun, s0, style='C0')
plt.show()


#Plots again
system_plot(mu_earthsun,L1,L2,L4,L5,xlim=[0.4,1.1],ylim=[-0.1,1],zoom=0,showSun=False,showL5=False)
s_end_orbit, t_orbit = shoot_to_poincare_y(mu_earthsun, s0, style='C0')

#Objective
s_obj = [s0_L4[0], s0_L4[1]]
def objective(ds, s0, s_obj, mu, style='none', step=0.01):
    s_fin, t = shoot_to_poincare_x(mu, [s0[0], s0[1], ds[0], ds[1]], style=style, crossings=2, bstep=step, x_obj=s0_L4[0])
    s_f = [s_fin[0], s_fin[1]]
    return np.array(s_f) - np.array(s_obj)

def to_minimize(halfway_point, style_1='none',style_2='none',show=False, step=0.05):
    s_halfway, t_half = shoot_to_poincare_x(mu_earthsun, s_transfer, style=style_1, crossings=1, x_obj=(L4+halfway_point), bstep=step)
    s0_halfway = [s_halfway[0], s_halfway[1]]
    ds_halfway = [s_halfway[2], s_halfway[3]]
    if show: print('Burn 1 at t: ' + str(t_half))
    success = False
    ds_guess = ds_halfway
    count=0
    while not(success) and count<11: # Retry until solution converges
        count+=1
        root1 = optimize.root(objective, ds_guess, args=(s0_halfway, s_obj, mu_earthsun,'none',step),method='hybr', tol=10**(-10))
        success = root1.success
        ds_guess = root1.x
    s_fin, t_1 = shoot_to_poincare_y(mu_earthsun, [s0_halfway[0],s0_halfway[1],root1.x[0],root1.x[1]], style=style_2, crossings=1, bstep=step, y_obj=s_obj[1])
    #if show: print(s_fin)
    deltav1 = np.array(root1.x)-np.array(ds_halfway)
    if show: print('Frist delta-v (halfway point): ' + str(deltav1) + ' [dx,dy] (absolute: ' + str(np.linalg.norm(deltav1)) + ')')
    deltav2 = np.array([s_fin[2],s_fin[3]]) - np.array([s0_L4[2],s0_L4[3]])
    if show: print('Second delta-v (L4):           ' + str(deltav2) + ' [dx,dy] (absolute: ' + str(np.linalg.norm(deltav2)) + ')')
    if show: print('Total transfer time: ' + str(t_half+t_1))
    #if not(success): return np.linalg.norm(deltav1)+np.linalg.norm(deltav2) + 100 # If the solution has not converged, the function will return an exagerated response so the solver does not take it as a valid solution
    return np.linalg.norm(deltav1)+np.linalg.norm(deltav2)


L4plusX = optimize.minimize_scalar(to_minimize, 0.3, bounds=(0.1,0.4),method='bounded',args=('none','none',False,0.02),options={'maxiter':7})

print('Optimal burn point found at x: ' + str(L4+L4plusX.x))

totalDv = to_minimize(L4plusX.x,style_1='C1',style_2='C2',show=True)
shoot_to_poincare_x(mu_earthsun, s0_L4, bstep=30, style='black')
print('For a total deltaV of: ' + str(totalDv))

legend_object11 = Patch(facecolor='black', edgecolor='black')
legend_object1 = Patch(facecolor='C0', edgecolor='black')
legend_object2 = Patch(facecolor='C1', edgecolor='black')
legend_object3 = Patch(facecolor='C2', edgecolor='black')
plt.legend(handles=[legend_object2, legend_object3, legend_object11],
           labels=['L1-L4 Manifold-contained path', 'Post-correction path', 'Final Planar L4 Orbit'],ncols=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='lower left')
plt.show()
