import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import math

###########################################################
# Initialisation
###########################################################
object_COM_x = 0
object_COM_y = 0

x1_original = 1
y1_original = 0
q1_1_original = 3*np.pi/4
q2_1_original = 0.2
q3_1_original = 3*np.pi/4

x2_original = 0
y2_original = 1
q1_2_original = 7*np.pi/4
q2_2_original = 0.2
q3_2_original = 7*np.pi/4

x1g_original = x1_original + q2_1_original*math.cos(q1_1_original)
y1g_original = y1_original + q2_1_original*math.sin(q1_1_original)
x2g_original = x2_original + q2_2_original*math.cos(q1_2_original)
y2g_original = y2_original + q2_2_original*math.sin(q1_2_original)
xc = 0
yc = 0
l = math.sqrt((xc-x1g_original)**2 + (yc-y1g_original)**2)
# print(f"l_const: {l}")

theta_const = math.atan2(y2g_original-y1g_original, x2g_original-x1g_original) - math.atan2(yc-y1g_original, xc-x1g_original)
# print(f"theta_const: {theta_const}")

###########################################################
# Problem Setup - Hyperparameters Setup
###########################################################

dt = 0.1
N = 10
T = 50

nu = 10 # [u1_1, u2_1, q1_1, q2_1, q3_1, u1_2, u2_2, q1_2, q2_2, q3_2]
nx = 10 # [x1, y1, q1_1, q2_1, q3_1, x2, y2, q1_2, q2_2, q3_2]

B = dt * np.eye(nx)

# X dot = B * U
t_grid = np.linspace(0, T*dt, T+N+1)
x_ref = np.linspace(0, T*dt, T+N+1) 
y_ref = np.tanh(t_grid)  # Example y reference

P = 20*np.array([[1, 0],
              [0, 1000]])
Q = 1
R = 0.1 * np.eye(nu)
R_ca = ca.SX(R)
Rd = 1.0 * np.eye(nu)
Rd_ca = ca.SX(Rd)

u_min, u_max = -1000, 1000
d = 0.5 # distance between two robots min threshold

###########################################################
# Placeholders
############################################################

x = ca.SX.sym('x', nx, N+1)                 # state trajectory
x_original = ca.SX.sym('x_o', nx, 1)        # state original position
u = ca.SX.sym('u', nu, N)                   # control trajectory

object_x = ca.SX.sym('object_x', 1, N+1)    # object COM trajectory
object_x_0 = ca.SX.sym('object_x_o', 1)     # object COM initial position
object_y = ca.SX.sym('object_y', 1, N+1)    # object COM trajectory 
object_y_0 = ca.SX.sym('object_y_o', 1)     # object COM initial position

object_next_x= ca.SX.sym('object_next_x',1, N+1)
object_next_y= ca.SX.sym('object_next_y',1, N+1)

ref_x_param = ca.SX.sym('ref_x', 1, N+1)         # reference x coordinates for object
ref_y_param = ca.SX.sym('ref_y', 1, N+1)         # reference y coordinates for object

# Compute gripper positions using lists (CasADi SX does not support assignment)
x_g1_list, y_g1_list, x_g2_list, y_g2_list = [], [], [], []
for i in range(N+1):
    x_g1_list.append(x[0, i] + x[3, i]*ca.cos(x[2, i]))
    y_g1_list.append(x[1, i] + x[3, i]*ca.sin(x[2, i]))
    x_g2_list.append(x[5, i] + x[8, i]*ca.cos(x[7, i]))
    y_g2_list.append(x[6, i] + x[8, i]*ca.sin(x[7, i]))

A_x = ca.vertcat(*x_g1_list)
A_y = ca.vertcat(*y_g1_list)
B_x = ca.vertcat(*x_g2_list)
B_y = ca.vertcat(*y_g2_list)
one  = ca.SX(1)
zero = ca.SX(0)

object_next_x[:,0]= object_x_0
object_next_y[:,0]= object_y_0
###############################################################
# Constraints: initial condition + dynamics
###############################################################
g = []


g.append(object_next_x[:,0]- object_x_0)
g.append(object_next_y[:,0]- object_y_0)
g.append(x[:, 0] - x_original[:, 0])  # initial condition
g.append(object_x[:, 0] - object_x_0)  # initial object position
g.append(object_y[:, 0] - object_y_0)  # initial object position

for k in range(N):
    # CONSTRAINT 1: 1st order integration: X_k+1 = X_k + dt * U_k
    x_next = x[:, k] + B @ u[:, k]
    g.append(x[:, k+1] - x_next)

    # CONSTRAINT 2
    # M. P = A_k+1, B_k+1
    M = ca.vertcat(
        ca.horzcat( A_x[k],  A_y[k], one, zero),
        ca.horzcat( A_y[k], -A_x[k], zero, one),
        ca.horzcat( B_x[k],  B_y[k], one, zero),
        ca.horzcat( B_y[k], -B_x[k], zero, one)
    )
    
    PARAM_T = ca.mtimes(ca.inv(M), ca.vertcat(A_x[k+1], A_y[k+1], B_x[k+1], B_y[k+1]))
    object_next_x[:,k+1] = ca.reshape(PARAM_T[0]*object_next_x[:,k] + PARAM_T[1]*object_next_y[:,k] + PARAM_T[2], 1, 1)
    object_next_y[:,k+1] = ca.reshape(PARAM_T[1]*object_next_x[:,k] - PARAM_T[0]*object_next_y[:,k] + PARAM_T[3], 1, 1)
    
    g.append(object_x[:,k+1] - object_next_x[:,k+1])
    g.append(object_y[:,k+1] - object_next_y[:,k+1])

    # Constraint 3: Distance constraint between two robots >=d
    dist_sq = (x[0, k] - x[5, k])*2 + (x[1, k] - x[6, k])*2
    g.append(dist_sq - d**2)

    # Constraint 4: let q3_1 = 0, q3_2 = 0
    g.append(x[4, k])
    g.append(x[9, k])


# print(type(M))
#################################################################
# Objective function
#################################################################
cost=0
for k in range(N):
    # Cost 1: track reference
    err = ca.vertcat(object_x[k+1] - ref_x_param[k+1], object_y[k+1] - ref_y_param[k+1])
    cost += ca.mtimes([err.T, P, err])

    # Cost 2: minimize control effort
    cost += ca.mtimes([u[:, k].T, R_ca, u[:, k]])

    # Cost 3: minimize distance between bots
    dist_err = ca.vertcat((x[0, k]-x[5, k]), (x[1, k]-x[6, k]))
    cost += ca.mtimes([dist_err.T, Q * ca.DM.eye(2), dist_err])

###############################################################
# Optimization problem setup
###############################################################
# print(object_y)
dec_vars = ca.vertcat(
    ca.reshape(x, -1, 1),
    ca.reshape(u, -1, 1),
    ca.reshape(object_x, -1, 1),
    ca.reshape(object_y, -1, 1)
)
g_vec = ca.vertcat(*g)
# print(g_vec)
ref_param_x = ca.reshape(ref_x_param, -1, 1)
ref_param_y = ca.reshape(ref_y_param, -1, 1)
x_original = ca.reshape(x_original, -1, 1) # Original State

# print(g_vec[2])

nlp = {'f': cost, 'x': dec_vars, 'g': g_vec, 'p': ca.vertcat(x_original, object_x_0, object_y_0, ref_param_x, ref_param_y)}
solver = ca.nlpsol('solver', 'ipopt', nlp)

###############################################################
# Bounds
################################################################
# Decision variable bounds: x, u, object_x, object_y
# X
x_min = np.array([-100, -100, -np.pi      , 0.1, 0    , -100, -100, -np.pi      , 0.1, 0])
x_max = np.array([ 100,  100,  np.pi- 1e-9, 2.0, np.pi,  100,  100,  np.pi- 1e-9, 2.0, np.pi])
# U
u_min = np.array([-1000, -1000, -np.pi/2, -0.5, -np.pi, -1000, -1000, -np.pi/2, -0.5, -np.pi])
u_max = np.array([ 1000,  1000,  np.pi/2,  0.5,  np.pi,  1000,  1000,  np.pi/2,  0.5,  np.pi])
# Object_x
object_x_min = np.array([-100])
object_x_max = np.array([ 100])
# Object_y
object_y_min = np.array([-100])
object_y_max = np.array([ 100])

lbx_1 = np.tile(x_min, N+1)
lbx_2 = np.tile(u_min, N)
lbx_3 = np.tile(object_x_min, N+1)
lbx_4 = np.tile(object_y_min, N+1)

ubx_1 = np.tile(x_max, N+1)
ubx_2 = np.tile(u_max, N)
ubx_3 = np.tile(object_x_max, N+1)
ubx_4 = np.tile(object_y_max, N+1)

lbx = np.concatenate([lbx_1, lbx_2, lbx_3, lbx_4]).tolist()
ubx = np.concatenate([ubx_1, ubx_2, ubx_3, ubx_4]).tolist()

# Constraint bounds
lbg = []
ubg = []

lbg += [0]
ubg += [0]

lbg += [0]
ubg += [0]

lbg += [0]*nx
ubg += [0]*nx

lbg += [0]
ubg += [0]

lbg += [0]
ubg += [0]

for k in range(N):
    # Constraint 1: system dynamics
    lbg += [0]*nx
    ubg += [0]*nx

    lbg += [0]
    ubg += [0]

    lbg += [0]
    ubg += [0]

    # Constraint 3: Distance constraint between two robots >=d
    lbg += [d**2]
    ubg += [ca.inf]

    # Constraint 4: let q3_1 = 0, q3_2 = 0
    lbg += [0]
    ubg += [0]

    lbg += [0]
    ubg += [0]

# print(len(ubg))
###############################################################
# Simulation
###############################################################

# Initial state
x0 = np.array([x1_original, y1_original, q1_1_original, q2_1_original, q3_1_original,
               x2_original, y2_original, q1_2_original, q2_2_original, q3_2_original])
x_current = x0
object_x_current = np.array([object_COM_x])
object_y_current = np.array([object_COM_y])


trajectory = [x_current.copy()] 
trajectory2 = [object_x_current.copy()]
controls = [] 
for t in range(T): # Set initial state and reference trajectory 
    print(t)
    x0 = x_current 
    objectx0 = object_x_current
    objecty0 = object_y_current
    # if t+N <= len(x_ref):
    #     ref_horizon = x_ref[t:t+N]
    # else:
    #     ref_horizon = np.concatenate([x_ref[t:], np.ones(t+N-len(x_ref))*x_ref[-1]])
    ref_horizon_x = x_ref[t+1:t+N+2]
    ref_horizon_y = y_ref[t+1:t+N+2]
    # Give the solver the states to work on
    init_guess = np.zeros((nx*(N+1) + nu*N +2*(N+1))) 
    
    print(t+N)
    sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=np.concatenate([x0, objectx0, objecty0, ref_horizon_x, ref_horizon_y])) 
    
    # Extract the optimal control input     
    sol_x = sol['x'].full().flatten() 
    u_opt = sol_x[nx*(N+1):nx*(N+1)+nu] 
    
    # Apply the first control input to the system 
    x_current = x_current + dt * u_opt[0] # simple Euler integration 
    object_x_current = sol_x[nx*(N+1)+nu*N: nx*(N+1)+nu*N+1]
    object_y_current = sol_x[nx*(N+1)+nu*N+ N: nx*(N+1)+ nu*N +N+1]
    trajectory2.append(object_y_current.copy())
    trajectory.append(x_current.copy()) 
    # print(f"trajectory: {trajectory}")
    controls.append(u_opt[0]) 
    
    # print(f"Time step {t}, State: {x_current}, Control: {u_opt[0]}") 

trajectory = np.array(trajectory) 
trajectory2 = np.array(trajectory2)
print(len(trajectory2))
########################################################### # Plotting results ########################### 
plt.figure() 
plt.plot(t_grid[0:T+1], y_ref[0:T+1], 'r--', label='Reference') 
plt.plot(t_grid[0:T+1], trajectory2[:,:], 'b-', label='State') 
plt.plot(t_grid[0:T+1], trajectory[:,1], 'g-', label='Bot2')
plt.plot(t_grid[0:T+1], trajectory[:,0], 'c-', label='Bot2_x')
plt.plot(t_grid[0:T+1], trajectory[:,6], 'y-', label='Bot1')
plt.xlabel('Time [s]') 
plt.ylabel('x') 
plt.legend() 
plt.show()