import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# System setup: Double integrator
# ------------------------------
dt = 0.1   # sampling time
N = 10     # horizon length
T = 100    # total simulation steps

nx = 2   # state dimension [x, v]
nu = 1   # input dimension [u]

A = np.array([[1, dt],
              [0, 1]])
B = np.array([[0.5*dt**2],
              [dt]])

# ------------------------------
# Reference trajectory
# ------------------------------
t_grid = np.linspace(0, T*dt, T+1)   # T+1 points including initial time
x_ref = np.sin(t_grid)

# ------------------------------
# Decision variables
# ------------------------------
x = ca.SX.sym('x', nx, N+1)      # state trajectory
u = ca.SX.sym('u', nu, N)        # control trajectory
x0_param = ca.SX.sym('x0', nx)   # measured initial state
ref_param = ca.SX.sym('ref', N)  # reference horizon

# ------------------------------
# Constraints: initial condition + dynamics
# ------------------------------
g = []
g.append(x[:,0] - x0_param)   # enforce initial state

for k in range(N):
    x_next = ca.mtimes(A, x[:,k]) + ca.mtimes(B, u[:,k])
    g.append(x[:,k+1] - x_next)

g = ca.vertcat(*g)   # (nx*(N+1),)

# ------------------------------
# Cost function
# ------------------------------
Q = np.diag([10, 1])   # state weights
R = np.diag([0.01])     # input weights

cost = 0
for k in range(N):
    x_err = ca.vertcat(x[0,k] - ref_param[k], x[1,k])
    cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u[:,k].T, R, u[:,k]])

# ------------------------------
# Bounds
# ------------------------------
x_min = np.array([-2.0, -3.0])
x_max = np.array([ 2.0,  3.0])
u_min = np.array([-1.0])
u_max = np.array([ 1.0])

lbx = (np.tile(x_min, N+1).tolist() + np.tile(u_min, N).tolist())
ubx = (np.tile(x_max, N+1).tolist() + np.tile(u_max, N).tolist())

# ------------------------------
# NLP problem
# ------------------------------
dec_vars = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))

nlp = {'x': dec_vars, 'f': cost, 'g': g, 'p': ca.vertcat(x0_param, ref_param)}

solver = ca.nlpsol('solver', 'ipopt', nlp,
                   {'ipopt.print_level': 0, 'print_time': 0})

# ------------------------------
# Simulation loop
# ------------------------------
x_current = np.array([0.0, 0.0])
trajectory = [x_current.copy()]
controls = []

for t in range(T):
    # reference horizon (pad with last value if needed)
    if t+N <= len(x_ref):
        ref_horizon = x_ref[t:t+N]
    else:
        ref_horizon = np.concatenate([x_ref[t:], np.ones(t+N-len(x_ref))*x_ref[-1]])

    p_val = np.concatenate([x_current, ref_horizon])

    # initial guess
    z0 = np.zeros(dec_vars.shape[0])

    sol = solver(x0=z0,
                 lbx=lbx, ubx=ubx,
                 lbg=np.zeros(g.shape[0]),
                 ubg=np.zeros(g.shape[0]),
                 p=p_val)

    z_opt = sol['x'].full().flatten()

    nxX = nx*(N+1)
    x_opt = z_opt[:nxX].reshape((nx, N+1), order='F')
    u_opt = z_opt[nxX:].reshape((nu, N), order='F')

    u_apply = u_opt[:,0]
    controls.append(u_apply)

    # simulate plant
    x_current = A.dot(x_current) + B.dot(u_apply)
    trajectory.append(x_current.copy())

trajectory = np.array(trajectory)    # shape (T+1, nx)
controls = np.array(controls)        # shape (T, nu)

# ------------------------------
# Plot results
# ------------------------------
plt.figure()
plt.plot(t_grid, x_ref, 'r--', label='Reference')
plt.plot(t_grid, trajectory[:,0], 'b-', label='Position')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Position')

plt.figure()
plt.step(t_grid[:-1], controls[:,0], 'g-', where='post', label='Control input')
plt.xlabel('Time [s]')
plt.ylabel('u')
plt.legend()
plt.show()
