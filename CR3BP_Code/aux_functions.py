# Aux. functions shared by many of the scripts developed for L1, L2, L4, and L5 motion.
# Aitor Urruticoechea 2022
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

#########################
# FUNCTION FROM https://github.com/RayleighLord/RayleighLordAnimations/blob/master/publication%20quality%20figures/fig_config.py
def add_grid(ax, lines=True, locations=None):
    """Add a grid to the current plot.
    Args:
        ax (Axis): axis object in which to draw the grid.
        lines (bool, optional): add lines to the grid. Defaults to True.
        locations (tuple, optional):
            (xminor, xmajor, yminor, ymajor). Defaults to None.
    """

    if lines:
        ax.grid(lines, alpha=0.5, which="minor", ls=":")
        ax.grid(lines, alpha=0.7, which="major")

    if locations is not None:

        assert (
            len(locations) == 4
        ), "Invalid entry for the locations of the markers"

        xmin, xmaj, ymin, ymaj = locations

        ax.xaxis.set_minor_locator(MultipleLocator(xmin))
        ax.xaxis.set_major_locator(MultipleLocator(xmaj))
        ax.yaxis.set_minor_locator(MultipleLocator(ymin))
        ax.yaxis.set_major_locator(MultipleLocator(ymaj))
###################


# Import basic data
def load_basic_data():
    try:
        basic_data = np.loadtxt('data/GMdata.txt')
    except:
        raise Exception('This code needs data on the G constant and the planetary masses!')
    G = basic_data[0]
    GM_sun = basic_data[1]
    GM_earth = basic_data[2]
    GM_moon = basic_data[3]
    mu_earthsun = (GM_earth + GM_moon)/(GM_sun + GM_earth + GM_moon)
    return mu_earthsun

# Import Lagrange Points
def import_LP():
    try:
        LagrangePoints = np.loadtxt('data/LagrangePoints.txt')
    except:
        raise Exception('This code needs to be run after the Lagrange Points have been calculated. Run lagrange_points.py first!')
    L1 = LagrangePoints[0]
    L2 = LagrangePoints[1]
    L4 = LagrangePoints[2]
    L5 = LagrangePoints[3]
    return L1,L2,L4,L5
    

# Function to plot CR3BP key points
def system_plot(mu_earthsun,L1,L2,L4,L5,xlim=[-0.05, 1.05],ylim=[-1, 1],zoom=0, showSun=True, showEarth=True, showL1=True, showL2=True, showL4=True, showL5=True):
    ax = plt.axes()
    if zoom==0:
        add_grid(ax, locations=(0.1,0.5,0.25,0.5))
    elif zoom==1:
        add_grid(ax, locations=(0.0025,0.01,0.0025,0.01))
    elif zoom==2:
        add_grid(ax, locations=(0.00025,0.001,0.00025,0.001))
    elif zoom==3:
        add_grid(ax, locations=(0.000025,0.0001,0.000025,0.0001))
    if showSun: plt.scatter(-mu_earthsun, 0, c='orange',edgecolor='black', label='Sun')
    if showEarth: plt.scatter(1-mu_earthsun,0, c='white',edgecolor='black', label='Earth')
    if showL1: plt.scatter(L1,0,c='#ff2f36', marker="x", label='L1')
    if showL2: plt.scatter(L2,0,c='#ac429e', marker="x", label='L2')
    if showL4: plt.scatter(L4,np.sqrt(3)/2,c='#febd00', marker="x", label='L4')
    if showL5: plt.scatter(L5,-np.sqrt(3)/2,c='#007dca', marker="x", label='L5')
    first_legend = plt.legend()
    plt.xlabel('x [NDU]')
    plt.ylabel('y [NDU]')
    try:
        plt.xlim(xlim)
        plt.ylim(ylim)
    except:
        pass
    ax.add_artist(first_legend)
    return ax

def Jacobi_c(s,mu):
    x = s[0]
    y = s[1]
    vx = s[2]
    vy = s[3]
    r = np.linalg.norm([x+mu-1, y]) #dist to smaller primary
    d = np.linalg.norm([x+mu, y]) #dist to larger primary
    omega = ((1-mu)/(d)) + (mu/(r)) + (1/2)*(x**2 + y**2)
    Jc = 2*omega - np.linalg.norm([vx,vy])**2
    return Jc

# CR3BP Movement description (2D)
def CR3BP_ds(s, mu):
    x = s[0]
    y = s[1]
    vx = s[2]
    vy = s[3]
    r = np.linalg.norm([(s[0])+mu-1, (s[1])]) #dist to smaller primary
    d = np.linalg.norm([(s[0])+mu, (s[1])]) #dist to larger primary
    ds = np.zeros(4)
    ds[0] = vx
    ds[1] = vy
    ds[2] = 2*vy + x - ((1-mu)/(d**3))*(x+mu) - (mu/(r**3))*(x-(1-mu))
    ds[3] = -2*vx + y - ((1-mu)/(d**3))*y - mu*y/(r**3)
    return ds

# CR3BP Propagation Function
def shoot(mu, s, tspan, style='none', t0=0, res=100):
    FUN = lambda t,x: np.array(CR3BP_ds(x,mu))
    sol = integrate.solve_ivp(FUN, [t0, tspan], s, t_eval = np.linspace(t0,tspan,res), method='DOP853', rtol = 1.e-10, atol = 1e-13)
    if style!='none':
        try:
            plt.plot(sol.y[0,:],sol.y[1,:],style)
            plt.pause(0.05)
        except:
            pass
    return sol

# Pointcare section (with x=k)
def shoot_to_poincare_x(mu, s_ini, bstep=3, style='none', crossings=2, x_obj=None):
    x = s_ini[0]
    y = s_ini[1]
    if x_obj == None: x0 = x
    else: x0 = x_obj
    dx = s_ini[2]
    dy = s_ini[3]
    t = 0
    n_cross = 0
    while n_cross < crossings:
        x_prev = x
        y_prev = y
        dx_prev = dx
        dy_prev = dy
        sol_t = shoot(mu, [x,y,dx,dy], t+bstep, style=style, t0=t,res=2)
        x = float(sol_t.y[0,1])
        y = float(sol_t.y[1,1])
        dx = float(sol_t.y[2,1])
        dy = float(sol_t.y[3,1])
        if (x_prev<x0 and x>x0) or (x_prev>x0 and x<x0):
            n_cross = n_cross+1
        t = t+bstep
    t = t-bstep
    x = x_prev
    y = y_prev
    dx = dx_prev
    dy = dy_prev
    def poincare_crossing(time):
        sol_t = shoot(mu, [x,y,dx,dy], time, t0=t,res=2, style='none')
        x_poincare = float(sol_t.y[0,1])
        return x_poincare - x0
    t_poincare = optimize.newton(poincare_crossing, t+bstep/2,maxiter=1000)
    sol_t = shoot(mu, [x,y,dx,dy], t_poincare, style=style, t0=t,res=2)
    x = float(sol_t.y[0,1])
    y = float(sol_t.y[1,1])
    dx = float(sol_t.y[2,1])
    dy = float(sol_t.y[3,1])
    s_end = [x,y,dx,dy]
    return s_end, t_poincare

# Pointcare section (with y=k)
def shoot_to_poincare_y(mu, s_ini, bstep=0.1, style='none',crossings=2, y_obj=None ):
    x = s_ini[0]
    y = s_ini[1]
    if y_obj == None: y0 = y
    else: y0 = y_obj
    dx = s_ini[2]
    dy = s_ini[3]
    t = 0
    n_cross = 0
    while n_cross < crossings:
        x_prev = x
        y_prev = y
        dx_prev = dx
        dy_prev = dy
        sol_t = shoot(mu, [x,y,dx,dy], t+bstep, style=style, t0=t,res=2)
        x = float(sol_t.y[0,1])
        y = float(sol_t.y[1,1])
        dx = float(sol_t.y[2,1])
        dy = float(sol_t.y[3,1])
        if (y_prev<y0 and y>y0) or (y_prev>y0 and y<y0):
            n_cross = n_cross+1
        t = t+bstep
    t = t-bstep
    x = x_prev
    y = y_prev
    dx = dx_prev
    dy = dy_prev
    def poincare_crossing(time):
        sol_t = shoot(mu, [x,y,dx,dy], time, t0=t,res=2, style='none')
        y_poincare = float(sol_t.y[1,1])
        return y_poincare - y0
    t_poincare = optimize.newton(poincare_crossing, t+bstep/2,maxiter=1000)
    sol_t = shoot(mu, [x,y,dx,dy], t_poincare, style=style, t0=t,res=2)
    x = float(sol_t.y[0,1])
    y = float(sol_t.y[1,1])
    dx = float(sol_t.y[2,1])
    dy = float(sol_t.y[3,1])
    s_end = [x,y,dx,dy]
    return s_end, t_poincare


# System definition (s' = A s) (2D)
def CR3BP_A(x,y,mu):
    r = +np.sqrt((x-1+mu)**2+y**2) #dist to smaller primary
    d = +np.sqrt((x+mu)**2+y**2) #dist to larger primary
    Uxx = 1 - (1-mu)/d**3 - mu/r**3 + (3*(1-mu)*(x+mu)**2)/d**5 +(3*mu*(x-1+mu)**2)/r**5
    Uxy = (3*(1-mu)*(x+mu)*y)/d**5 + (3*mu*(x-1+mu)*y)/r**5
    Uyx = Uxy
    Uyy = 1 - (1-mu)/d**3 - mu/r**3 + (3*(1-mu)*y**2)/d**5 + (3*mu*y**2)/r**5
    A = [[0, 0, 1, 0],[0, 0, 0, 1],[Uxx, Uxy, 0, 2],[Uyx, Uyy, -2, 0]]
    return A

# Numerical processes often times leave behind very small results that can be considered zero
def clear_small(matrix,tol):
    if np.ndim(matrix)==2: #CASE 2D
        for n in range(np.size(matrix,0)):
            for m in range(np.size(matrix,1)):
                real = np.real(matrix[n,m])
                imag = np.imag(matrix[n,m])
                if real<tol and real>-tol:
                    matrix[n,m] = np.imag(matrix[n,m])*1j
                if imag<tol and imag>-tol:
                    matrix[n,m] = np.real(matrix[n,m])
    else: #CASE 1D
        for n in range(np.size(matrix,0)):
            real = np.real(matrix[n])
            imag = np.imag(matrix[n])
            if real<tol and real>-tol:
                matrix[n] = imag*1j
            if imag<tol and imag>-tol:
                matrix[n] = real
    return matrix

# Remove imaginary part from eignvectors to create C matrix
def remove_i(veps):
    C = np.zeros([np.size(veps,0),np.size(veps,1)])
    for a in range(int(np.size(veps,0)/2)):
        C[:,2*a] = np.real(veps[:,2*a])
        C[:,2*a+1] = np.imag(veps[:,2*a])
    return C

# First set of lineal equations for collinear points
def lin_eq_c1(vaps, A1, A2, y0):
    lam = np.real(vaps[0])
    c = (lam**2-1)/(2*lam)
    def y_fun(t):
        return c*A1*np.e**(lam*t) + c*A2*np.e**(-lam*t) - y0
    t0 = optimize.fsolve(y_fun,0.1)
    x = A1*np.e**(lam*t0) + A2*np.e**(-lam*t0)
    dx = (A1/lam) *np.e**(lam*t0) - (A2/lam) *np.e**(-lam*t0)
    dy = (c*A1/lam) *np.e**(lam*t0) - (c*A2/lam) * np.e**(-lam*t0)
    return [t0, x, dx, dy]

# Second set of lineal equations for collinear points
def lin_eq_c2(vaps, A3, A4, y0):
    w1 = np.imag(vaps[2])
    kappa = -(w1**2+1)/(2*w1)
    def y_fun(t):
        return kappa*A3*np.cos(w1*t) + kappa*A4*np.sin(w1*t) - y0
    t0 = optimize.fsolve(y_fun, 0)
    x = A3*np.cos(w1*t0) + A4*np.sin(w1*t0)
    dx = -A3*w1*np.sin(w1*t0) +  A4*w1*np.cos(w1*t0)
    dy = -kappa*A3*w1*np.sin(w1*t0) +  kappa*A4*w1*np.cos(w1*t0)
    return [t0, x, dx, dy]

# First set of lineal equations for equilateral points
def lin_eq_1(C, mvaps, A1, A2, y0):
    # A3 = A4 = 0
    Vw1_0 = C[0,0]
    Vw1_1 = C[1,0]
    Vw1_2 = C[2,0]
    Vw1_3 = C[3,0]
    Uw1_0 = C[0,1]
    Uw1_1 = C[1,1]
    Uw1_2 = C[2,1]
    Uw1_3 = C[3,1]
    w1 = mvaps[0]
    #w2 = mvaps[2]
    def y_fun(t):
        return A1*np.cos(w1*t)*Vw1_1 + A2*np.sin(w1*t)*Uw1_1 - y0
    t0 = optimize.fsolve(y_fun,0.1)
    x = A1*np.cos(w1*t0)*Vw1_0 + A2*np.sin(w1*t0)*Uw1_0
    dx = A1*np.cos(w1*t0)*Vw1_2 + A2*np.sin(w1*t0)*Uw1_2
    dy = A1*np.cos(w1*t0)*Vw1_3 + A2*np.sin(w1*t0)*Uw1_3
    return [t0, x, dx, dy]

# Second set of Lineal equations for equilateral points
def lin_eq_2(C, mvaps, A3, A4, y0):
    # A1 = A2 = 0
    Vw2_0 = C[0,2]
    Vw2_1 = C[1,2]
    Vw2_2 = C[2,2]
    Vw2_3 = C[3,2]
    Uw2_0 = C[0,3]
    Uw2_1 = C[1,3]
    Uw2_2 = C[2,3]
    Uw2_3 = C[3,3]
    #w1 = mvaps[0]
    w2 = mvaps[2]
    def y_fun(t):
        return A3*np.cos(w2*t)*Vw2_1 + A4*np.sin(w2*t)*Uw2_1 - y0
    t0 = optimize.fsolve(y_fun,0.1)
    x = A3*np.cos(w2*t0)*Vw2_0 + A4*np.sin(w2*t0)*Uw2_0
    dx = A3*np.cos(w2*t0)*Vw2_2 + A4*np.sin(w2*t0)*Uw2_2
    dy = A3*np.cos(w2*t0)*Vw2_3 + A4*np.sin(w2*t0)*Uw2_3
    return [t0, x, dx, dy]

# Variationals - STM
def variationals(x, mu):
    s = [x[0], x[1], x[2], x[3]]
    ds = CR3BP_ds(s, mu)
    r = +np.sqrt((x[0]-1+mu)**2+x[1]**2) #dist to smaller primary
    d = +np.sqrt((x[0]+mu)**2+x[1]**2) #dist to larger primary
    Uxx = 1 - (1-mu)/d**3 - mu/r**3 + (3*(1-mu)*(x[0]+mu)**2)/d**5 +(3*mu*(x[0]-1+mu)**2)/r**5
    Uxy = (3*(1-mu)*(x[0]+mu)*x[1])/d**5 + (3*mu*(x[0]-1+mu)*x[1])/r**5
    Uyx = Uxy
    Uyy = 1 - (1-mu)/d**3 - mu/r**3 + (3*(1-mu)*x[1]**2)/d**5 + (3*mu*x[1]**2)/r**5
    lam_d = np.array([
        [x[6]                     , x[10]                     , x[14]                       , x[18]],
        [x[7]                     , x[11]                     , x[15]                       , x[19]],
        [Uxx*x[4]+Uxy*x[5]+2*x[7] , Uxx*x[8]+Uxy*x[9]+2*x[11] , Uxx*x[12]+Uxy*x[13]+2*x[15] , Uxx*x[16]+Uxy*x[17]+2*x[19]],
        [Uyx*x[4]+Uyy*x[5]-2*x[6] , Uyx*x[8]+Uyy*x[9]-2*x[10] , Uyx*x[12]+Uyy*x[13]-2*x[14] , Uyx*x[16]+Uyy*x[17]-2*x[18]],
    ])
    var = np.append(ds, lam_d[:,0])
    var = np.append(var, lam_d[:,1])
    var = np.append(var, lam_d[:,2])
    var = np.append(var, lam_d[:,3])
    return var

def variationals_prop(mu, s0, tspan, t0=0):
    tspan = round(tspan, 5)
    FUN = lambda t, x: np.array(variationals(x, mu))
    init = np.append(np.array(s0), np.identity(4))
    sol = integrate.solve_ivp(FUN, [t0, tspan], init, t_eval = np.linspace(t0,tspan,int(tspan*100000)), method='DOP853',rtol = 1.e-10, atol = 1e-13)
    STM = sol.y[:,-1]
    phi = np.zeros([4,4])
    for n in range(4):
        phi[n,0] = STM[4+4*n]
        phi[n,1] = STM[4+4*n+1]
        phi[n,2] = STM[4+4*n+2]
        phi[n,3] = STM[4+4*n+3]
    return phi


