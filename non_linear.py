import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters
nx = 6       # states: [xp, yp, l1, theta1, xb, yb]
nu = 4       # inputs: [xb_dot, yb_dot, l1_dot, theta1_dot]
N = 50       # horizon
Ts = 0.1     # timestep

# --- Reference trajectory
x_ref_traj = 5*np.linspace(0, 5, N+1)
y_ref_traj = 5*np.linspace(0, 5, N+1)

# --- Decision variables
x = ca.SX.sym('x', nx, N+1)
u = ca.SX.sym('u', nu, N)

# --- Initial state: xp, yp, l1, theta1, xb, yb
x0 = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0])

# --- Cost matrices
Q = np.diag([10, 10])  # tracking weight on xp, yp
R = np.eye(nu) * 0.1   # input weight

# --- Objective and constraints
cost = 0
constraints = []

for k in range(N):
    xk = x[:, k]
    uk = u[:, k]

    # --- Explicit next-state update
    x_next = ca.vertcat(
        xk[0] + ca.cos(xk[3])*uk[2] - xk[2]*ca.sin(xk[3])*uk[3],  # xp
        xk[1] + ca.sin(xk[3])*uk[2] + xk[2]*ca.cos(xk[3])*uk[3],  # yp
        xk[2] + uk[2],  # l1
        xk[3] + uk[3],  # theta1
        xk[4] + uk[0],  # xb
        xk[5] + uk[1]   # yb
    )

    # --- Dynamics equality
    constraints.append(x[:, k+1] - x_next)

    # --- Tracking cost
    x_ref = ca.vertcat(x_ref_traj[k], y_ref_traj[k])
    cost += ca.mtimes([(xk[0:2] - x_ref).T, Q, (xk[0:2] - x_ref)]) + ca.mtimes([uk.T, R, uk])

# --- Flatten constraints and variables
g = ca.vertcat(*constraints)
opt_vars = ca.vertcat(ca.reshape(x, -1,1), ca.reshape(u, -1,1))

# --- Bounds
lbx = []
ubx = []

for k in range(N+1):
    # xp, yp: no bounds
    lbx += [-ca.inf, -ca.inf]
    ubx += [ ca.inf,  ca.inf]

    # l1
    lbx += [0.2]
    ubx += [2.2]

    # theta1
    lbx += [0]
    ubx += [2*np.pi]

    # xb, yb: no bounds
    lbx += [-ca.inf, -ca.inf]
    ubx += [ ca.inf,  ca.inf]

# input bounds
for k in range(N):
    lbx += [-1.0]*nu
    ubx += [ 1.0]*nu

# constraints bounds (dynamics equality)
lbg = [0.0]*(nx*N)
ubg = [0.0]*(nx*N)

# --- Initial guess
x_init = np.tile(x0.reshape(-1,1), N+1)
u_init = np.zeros((nu,N))
init_guess = np.concatenate([x_init.flatten(), u_init.flatten()])

# --- NLP solver
nlp = {'x': opt_vars, 'f': cost, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# --- Solve NLP
sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
sol_vars = sol['x'].full().flatten()

# --- Extract solution
x_sol = sol_vars[:nx*(N+1)].reshape((nx, N+1))
u_sol = sol_vars[nx*(N+1):].reshape((nu, N))

# --- Plot trajectory
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(
    x_sol[4,:] + x_sol[2,:]*np.cos(x_sol[3,:]),
    x_sol[5,:] + x_sol[2,:]*np.sin(x_sol[3,:]),
    'g-', label='tip trajectory'
)
plt.plot(x_ref_traj, y_ref_traj, '--', label='reference')
plt.title("Trajectory Tracking")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(1,2,2)
plt.step(range(N), u_sol[0,:], where='post', label='xb_dot')
plt.step(range(N), u_sol[1,:], where='post', label='yb_dot')
plt.step(range(N), u_sol[2,:], where='post', label='l1_dot')
plt.step(range(N), u_sol[3,:], where='post', label='theta1_dot')
plt.title("Inputs")
plt.xlabel("Timestep")
plt.legend()
plt.show()
