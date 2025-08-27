import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# System dynamics (double integrator: position + velocity)
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[0.5],
              [1]])

# Cost function weights
Q = np.diag([10, 1])   # penalize position error more than velocity
R = np.eye(1) * 0.1

# Horizon
N = 10
T = 40

# Initial state (flat array, shape (2,))
x = np.array([0.0, 0.0])
x_traj = [x]
u_traj = []

# Reference trajectory (sine wave)
ref_pos = np.sin(np.linspace(0, 10, T+N))  # sine position reference
ref_vel = np.gradient(ref_pos)             # approximate desired velocity
x_ref = np.vstack((ref_pos, ref_vel))      # shape (2, T+N)

for t in range(T):
    # Decision variables
    u = cp.Variable((1, N))
    x_var = cp.Variable((2, N+1))

    # Cost and constraints
    cost = 0
    constraints = [x_var[:, 0] == x]  # initial condition matches current state
    for k in range(N):
        x_ref_k = x_ref[:, t+k]  # desired state at time k
        cost += cp.quad_form(x_var[:, k] - x_ref_k, Q) + cp.quad_form(u[:, k], R)
        constraints += [x_var[:, k+1] == A @ x_var[:, k] + B @ u[:, k]]
        constraints += [cp.abs(u[:, k]) <= 1]  # input constraint

    # Terminal cost
    cost += cp.quad_form(x_var[:, N] - x_ref[:, t+N], Q)

    # Solve MPC optimization
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    # Apply first control
    u_applied = u[:, 0].value.item()
    x = (A @ x + B.flatten() * u_applied).flatten()
    x_traj.append(x)
    u_traj.append(u_applied)

x_traj = np.vstack(x_traj)

# --- Plot Results ---
plt.figure()
plt.plot(ref_pos[:T+1], "r--", label="Reference Position")
plt.plot(x_traj[:, 0], "b", label="MPC Position")
plt.legend()
plt.title("MPC Trajectory Tracking - Position")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.show()

plt.figure()
plt.plot(ref_vel[:T+1], "r--", label="Reference Velocity")
plt.plot(x_traj[:, 1], "b", label="MPC Velocity")
plt.legend()
plt.title("MPC Trajectory Tracking - Velocity")
plt.xlabel("Time step")
plt.ylabel("Velocity")
plt.show()

plt.figure()
plt.step(range(T), u_traj, where="post")
plt.title("MPC Control Input")
plt.xlabel("Time step")
plt.ylabel("u")
plt.show()
