import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import math

###########################################################
# Initialisation
###########################################################
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
print(f"l_const: {l}")

theta_const = math.atan2(y2g_original-y1g_original, x2g_original-x1g_original) - math.atan2(yc-y1g_original, xc-x1g_original)
print(f"theta_const: {theta_const}")

###########################################################
# Problem Setup
###########################################################

dt = 0.1
N = 10
T = 20

nu = 10
nx = 10

# A = np.eye(nx)
B = dt * np.eye(nx)  # Each state updated by its own control

t_grid = np.linspace(0, T*dt, T+N+1)
x_ref = np.linspace(0, T*dt, T+N+1)
# y_ref = 5*np.linspace(0, T*dt, T+N+1)
y_ref = np.tanh(t_grid)  # Example y reference

#2 x 2 matrix for (x_c, y_c) tracking cost
P = 20*np.array([[10, 0],
              [0, 0.9]])
Q = 1
R = 0.1 * np.eye(nu)
R_ca = ca.SX(R)
Rd = 1.0 * np.eye(nu)
Rd_ca = ca.SX(Rd)

u_min, u_max = -1000, 1000
d = 0.6

##############################################################
# Placeholders
##############################################################
x = ca.SX.sym('x', nx, N+1)
u = ca.SX.sym('u', nu, N)
x0_param = ca.SX.sym('x0', nx)
ref_param_x = ca.SX.sym('ref_x', N)
ref_param_y = ca.SX.sym('ref_y', N)

# Compute gripper positions using lists (CasADi SX does not support assignment)
x_g1_list, y_g1_list, x_g2_list, y_g2_list = [], [], [], []
for i in range(N+1):
    x_g1_list.append(x[0, i] + x[3, i]*ca.cos(x[2, i]))
    y_g1_list.append(x[1, i] + x[3, i]*ca.sin(x[2, i]))
    x_g2_list.append(x[5, i] + x[8, i]*ca.cos(x[7, i]))
    y_g2_list.append(x[6, i] + x[8, i]*ca.sin(x[7, i]))
x_g1 = ca.vertcat(*x_g1_list)
print("x_g1:", x_g1)
y_g1 = ca.vertcat(*y_g1_list)
x_g2 = ca.vertcat(*x_g2_list)
y_g2 = ca.vertcat(*y_g2_list)

# Compute object positions using lists
x_c_list, y_c_list = [], []
for i in range(N+1):
    angle = ca.atan2(y_g2[i]-y_g1[i], x_g2[i]-x_g1[i]) - theta_const
    x_c_list.append(l * ca.cos(angle) + x_g1[i])
    y_c_list.append(l * ca.sin(angle) + y_g1[i])
x_c = ca.vertcat(*x_c_list)
y_c = ca.vertcat(*y_c_list)

###############################################################
# Constraints: initial condition + dynamics
###############################################################
g = []

g.append(x[:, 0] - x0_param)
g.append(x_c[0] - xc)
g.append(y_c[0] - yc)
for k in range(N):
    # Dynamics
    g.append(x[:, k+1] - (x[:, k] + ca.mtimes(B, u[:, k])))

    # Distance between bots >= d
    g.append(((x[0, k]-x[5, k])**2 + (x[1, k]-x[6, k])**2) - d**2)

    # # Input bounds
    g.append(u[:, k] - u_min)
    g.append(u_max - u[:, k])

    # Gripper angle constraints
    g.append((ca.atan2((y_g2[k] - y_g1[k]), (x_g2[k] - x_g1[k])) - x[2, k] - x[4, k]))
    g.append((ca.atan2((y_g1[k] - y_g2[k]), (x_g1[k] - x_g2[k])) - x[7, k] - x[9, k]))

    # Remove hard constraint on x_c, use cost instead
    # g.append(x_c[k] - ref_param_x[k])

    # Distance between grippers = constant l
    g.append(((x_g2[k]-x_g1[k])**2 + (y_g2[k]-y_g1[k])**2) - l**2)

    # let q3_1 = 0, q3_2 = 0
    g.append(x[4, k])
    g.append(x[9, k]) 

################################################################
# Cost function
################################################################
cost = 0
for k in range(N):
    # Trajectory tracking cost (use both x and y reference)
    x_err = ca.vertcat((x_c[k] - ref_param_x[k]), (y_c[k] - ref_param_y[k]))

    # cost += P * ca.mtimes([x_err.T, x_err])
    cost += ca.mtimes([x_err.T, P, x_err])

    # Minimise distance between bots
    dist_err = ca.vertcat((x[0, k]-x[5, k]), (x[1, k]-x[6, k]))
    cost += Q * ca.mtimes([dist_err.T, dist_err])

    # cost += ca.mtimes([u[:, k].T, R_ca, u[:, k]])

    # if k > 0:
    #     du = u[:, k] - u[:, k-1]
    #     cost += ca.mtimes([du.T, Rd_ca, du])

###########################################################
dec_vars = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))
g_vec = ca.vertcat(*g)
nlp = {'f': cost, 'x': dec_vars, 'g': g_vec, 'p': ca.vertcat(x0_param, ref_param_x, ref_param_y)}
solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0, "ipopt.tol": 1e-8, "ipopt.constr_viol_tol": 1e-8})

############################################################
# Bounds
############################################################

x_min = np.array([-np.inf, -np.inf, -np.pi, 0.1, -np.pi,
                  -np.inf, -np.inf, -np.pi, 0.1, -np.pi], dtype=float)
x_max = np.array([ np.inf,  np.inf,  np.pi - 1e-9, 2.0,  np.pi - 1e-9,
                   np.inf,  np.inf,  np.pi - 1e-9, 2.0,  np.pi - 1e-9], dtype=float)

u_min_arr = u_min * np.ones(nu)
u_max_arr = u_max * np.ones(nu)

lbx_states = np.tile(x_min, N+1)
ubx_states = np.tile(x_max, N+1)
lbx_inputs = np.tile(u_min_arr, N)
ubx_inputs = np.tile(u_max_arr, N)

lbx = np.concatenate([lbx_states, lbx_inputs]).tolist()
ubx = np.concatenate([ubx_states, ubx_inputs]).tolist()

lbg = []
ubg = []

lbg += [0]*nx
ubg += [0]*nx

lbg += [0]
ubg += [0]

lbg += [0]
ubg += [0]

for k in range(N):
    lbg += [0]*nx
    ubg += [0]*nx

    lbg += [0]
    ubg += [ca.inf]

    lbg += [0]*nu
    ubg += [ca.inf]*nu
    lbg += [0]*nu
    ubg += [ca.inf]*nu

    lbg += [0]*2
    ubg += [0]*2

    lbg += [0]
    ubg += [0]

    # lbg += [0]
    # ubg += [0]

    lbg += [0]*2
    ubg += [0]*2

###########################################################
# Simulation
###########################################################

x_current = np.array([x1_original, y1_original, q1_1_original, q2_1_original, q3_1_original,
                      x2_original, y2_original, q1_2_original, q2_2_original, q3_2_original])
trajectory = [x_current.copy()]
controls = []

for t in range(T):
    print(t)
    x0 = x_current
    if t+N <= len(x_ref):
        ref_horizon_x = x_ref[t:t+N]
        ref_horizon_y = y_ref[t:t+N]
    else:
        ref_horizon_x = np.pad(x_ref[t:], (0, (t+N) - len(x_ref)), mode='edge')
        ref_horizon_y = np.pad(y_ref[t:], (0, (t+N) - len(y_ref)), mode='edge')

    init_guess = np.tile(x_current, N+1)
    init_guess = np.concatenate([init_guess, np.zeros(nu*N)])

    sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg,
                 p=np.concatenate([x0, ref_horizon_x, ref_horizon_y]))

    sol_x = sol['x'].full().flatten()
    u_opt = sol_x[nx*(N+1):nx*(N+1)+nu]
    # x_current = A.dot(x_current) + B.dot(u_apply)
    # Proper state update for all states
    x_current += B.dot(u_opt)  # Euler integration for all states
    trajectory.append(x_current.copy())
    controls.append(u_opt)

trajectory = np.array(trajectory)
controls = np.array(controls)

x1g = trajectory[:, 0] + trajectory[:, 3]*np.cos(trajectory[:, 2])
y1g = trajectory[:, 1] + trajectory[:, 3]*np.sin(trajectory[:, 2])
x2g = trajectory[:, 5] + trajectory[:, 8]*np.cos(trajectory[:, 7])
y2g = trajectory[:, 6] + trajectory[:, 8]*np.sin(trajectory[:, 7])
xc_traj = x1g + l * np.cos(np.arctan2(y2g-y1g, x2g-x1g) - theta_const)
yc_traj = y1g + l * np.sin(np.arctan2(y2g-y1g, x2g-x1g) - theta_const)

np.savetxt('xc_traj.txt', xc_traj)
np.savetxt('yc_traj.txt', yc_traj)
np.savetxt('x1g.txt', x1g)
np.savetxt('y1g.txt', y1g)
np.savetxt('x2g.txt', x2g)
np.savetxt('y2g.txt', y2g)

###########################################################
# Plotting results
###########################################################
print(trajectory[:, :])

# plt.figure()
# plt.plot(t_grid, x_ref, 'r--', label='Reference x')
# plt.plot(t_grid, y_ref, 'g--', label='Reference y')
# # plt.plot(t_grid, trajectory[:, 0], 'b-', label='State x_1')
# # plt.plot(t_grid, trajectory[:, 1], 'c-', label='State y_1')
# # plt.plot(t_grid, trajectory[:, 2], 'm-', label='State q1_1')
# # plt.plot(t_grid, trajectory[:, 4], 'y-', label='State q3_1')
# plt.plot(t_grid, yc_traj, 'k-', label='y_c')
# plt.plot(t_grid, xc_traj, 'orange', label='x_c')
# plt.xlabel('Time [s]')
# plt.ylabel('States')
# plt.legend()
# plt.show()

plt.figure()
plt.plot(x_ref, y_ref, 'm--', label='Reference trajectory')
plt.plot(xc_traj, yc_traj, 'b-', label='End-effector trajectory')
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r--', label='Bot 1 trajectory')
plt.plot(trajectory[:, 5], trajectory[:, 6], 'g--', label='Bot 2 trajectory')
plt.legend()
plt.show()