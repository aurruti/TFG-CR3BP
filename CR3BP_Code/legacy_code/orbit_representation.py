# Orbit .txt reader
# Aitor Urruticoechea 2022
from aux_functions import *
import numpy as np

mu_earthsun = load_basic_data()
L1, L2, L4, L5 = import_LP()

choice = input('Orbit around: ')

system_plot(mu_earthsun,L1,L2,L4,L5) 
if choice == 'L4':
    print("_________________________________________________________________________________________________")
    print("") 
    print("                                     L4 Planar Orbits")
    print("_________________________________________________________________________________________________")
    L4_planar_1 = np.loadtxt('data/L4_PlanarOrbits_1.txt')
    L4_planar_2 = np.loadtxt('data/L4_PlanarOrbits_2.txt')
    count = 0
    for orbit in L4_planar_1:
        count = count+1
        print('Type-1 Orbit nº' + str(count) + ' at y0 = '+ str(orbit[0]) + ' | dx = ' + str(orbit[1]) + ' | dy = ' + str(orbit[2]) + '    (all in NDU)')
        shoot(mu_earthsun, [L4, orbit[0], orbit[1], orbit[2]],tspan=10000,res=100000)
    count = 0
    for orbit in L4_planar_2:
        count = count+1
        print('Type-2 Orbit nº' + str(count) + ' at y0 = '+ str(orbit[0]) + ' | dx = ' + str(orbit[1]) + ' | dy = ' + str(orbit[2]) + '    (all in NDU)')
        shoot(mu_earthsun, [L4, orbit[0], orbit[1], orbit[2]],tspan=10000,res=100000)
elif choice == 'L5':
    print("_________________________________________________________________________________________________")
    print("") 
    print("                                     L5 Planar Orbits")
    print("_________________________________________________________________________________________________")
    L5_planar_1 = np.loadtxt('data/L5_PlanarOrbits_1.txt')
    L5_planar_2 = np.loadtxt('data/L5_PlanarOrbits_2.txt')
    count = 0
    for orbit in L5_planar_1:
        count = count+1
        print('Type-1 Orbit nº' + str(count) + ' at y0 = '+ str(orbit[0]) + ' | dx = ' + str(orbit[1]) + ' | dy = ' + str(orbit[2]) + '    (all in NDU)')
        shoot(mu_earthsun, [L5, orbit[0], orbit[1], orbit[2]],tspan=10000,res=100000)
    count = 0
    for orbit in L5_planar_2:
        count = count+1
        print('Type-2 Orbit nº' + str(count) + ' at y0 = '+ str(orbit[0]) + ' | dx = ' + str(orbit[1]) + ' | dy = ' + str(orbit[2]) + '    (all in NDU)')
        shoot(mu_earthsun, [L5, orbit[0], orbit[1], orbit[2]],tspan=10000,res=100000)    
plt.show()
print('')
print('')
input('ENTER to exit')
