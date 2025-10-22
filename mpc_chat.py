# mpc_fixed_with_tests.py
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# -------------------------
# Unit tests (transform & distance constraint)
# -------------------------
def solve_transform_numpy(Ax_k, Ay_k, Bx_k, By_k, Ax_k1, Ay_k1, Bx_k1, By_k1, reg=1e-9):
    """
    Build the same small 4x4 linear system as in the MPC and solve for [a,b,tx,ty]
    using numpy (for unit testing).
    """
    M = np.vstack([
        np.hstack([Ax_k, Ay_k, 1.0, 0.0]),
        np.hstack([Ay_k, -Ax_k, 0.0, 1.0]),
        np.hstack([Bx_k, By_k, 1.0, 0.0]),
        np.hstack([By_k, -Bx_k, 0.0, 1.0])
    ])
    b = np.array([Ax_k1, Ay_k1, Bx_k1, By_k1])
    # regularize if near singular
    M_reg = M + reg * np.eye(4)
    sol = np.linalg.solve(M_reg, b)
    return sol  # [a, b, tx, ty]

def unit_test_transform():
    # ground-truth transform: rotation theta and translation tx,ty
    theta = 0.37
    a_true = math.cos(theta)
    b_true = math.sin(theta)
    tx_true = 0.55
    ty_true = -0.12

    # pick two distinct gripper points (in world coordinates at time k)
    A_k = np.array([1.23, -0.5])
    B_k = np.array([-0.3, 0.87])

    # apply transform p_{k+1} = R p_k + t
    R = np.array([[a_true, -b_true],[b_true, a_true]])
    t = np.array([tx_true, ty_true])
    A_k1 = (R @ A_k) + t
    B_k1 = (R @ B_k) + t

    sol = solve_transform_numpy(A_k[0], A_k[1], B_k[0], B_k[1], A_k1[0], A_k1[1], B_k1[0], B_k1[1])
    a_est, b_est, tx_est, ty_est = sol

    ok = np.allclose([a_est,b_est,tx_est,ty_est], [a_true,b_true,tx_true,ty_true], atol=1e-7)
    print("Transform unit test:", "PASS" if ok else "FAIL")
    if not ok:
        print(" estimated:", sol, " true:", [a_true,b_true,tx_true,ty_true])
    return ok

def unit_test_distance():
    # distance constraint: dist_sq - d^2 >= 0
    d = 0.5
    # case 1: exactly at distance d
    x1 = np.array([0.0, 0.0])
    x2 = np.array([d, 0.0])
    dist_sq = np.sum((x1 - x2)**2)
    residual = dist_sq - d**2
    ok1 = abs(residual) < 1e-9

    # case 2: less than d (violation)
    x2b = np.array([0.4*d, 0.0])
    residual2 = np.sum((x1-x2b)**2) - d**2
    ok2 = residual2 < 0

    print("Distance unit test:", "PASS" if (ok1 and ok2) else "FAIL")
    return ok1 and ok2

if __name__ == "__main__":
    # run unit tests first
    t1 = unit_test_transform()
    t2 = unit_test_distance()
    if not (t1 and t2):
        print("Unit tests failed - stop. Fix unit test failures first.")
    else:
        print("All unit tests passed. Proceeding to run MPC simulation...")


# -------------------------
# MPC script (fixed)
# -------------------------
# (If running tests above failed, the script will still continue — remove that if you want strong gating.)

# Initial positions and states
object_COM_x = 0.0
object_COM_y = 0.0

x1_original = 1.0
y1_original = 0.0
q1_1_original = 3*np.pi/4
q2_1_original = 0.2
q3_1_original = 3*np.pi/4

x2_original = 0.0
y2_original = 1.0
q1_2_original = 7*np.pi/4
q2_2_original = 0.2
q3_2_original = 7*np.pi/4

# some derived gripper positions at initialization (not strictly required, keep for intuition)
x1g_original = x1_original + q2_1_original * math.cos(q1_1_original)
y1g_original = y1_original + q2_1_original * math.sin(q1_1_original)
x2g_original = x2_original + q2_2_original * math.cos(q1_2_original)
y2g_original = y2_original + q2_2_original * math.sin(q1_2_original)

xc = 0.0
yc = 0.0

l = math.sqrt((xc-x1g_original)**2 + (yc-y1g_original)**2)
theta_const = math.atan2(y2g_original-y1g_original, x2g_original-x1g_original) - math.atan2(yc-y1g_original, xc-x1g_original)

# Hyperparameters
dt = 0.1
N = 5              # MPC horizon
print(f"N: {N}")
T = 100           # total simulation steps

nu = 10 # dimension of u per step (same as nx here)
nx = 10 # dimension of state

B = dt * np.eye(nx)

t_grid = np.linspace(0, T*dt, T+1)
x_ref = np.linspace(0, T*dt, T+1)          # example x reference (unused beyond demo)
y_ref =4*np.cos(t_grid)                    # example y reference

P = np.array([[20.0, 0.0],[0.0, 20.0]])
Q = 1.0
R = 0.1 * np.eye(nu)
R_ca = ca.SX(R)

d = 0.5 # min distance

# -------------------------
# CasADi symbolic variables
# -------------------------
x = ca.SX.sym('x', nx, N+1)                   # states over horizon
x_original = ca.SX.sym('x_o', nx, 1)          # initial state parameter
u = ca.SX.sym('u', nu, N)                     # control over horizon

object_x = ca.SX.sym('object_x', 1, N+1)      # object x decision vars
object_y = ca.SX.sym('object_y', 1, N+1)      # object y decision vars
object_x_0 = ca.SX.sym('object_x_o', 1)       # object initial param
object_y_0 = ca.SX.sym('object_y_o', 1)

ref_x_param = ca.SX.sym('ref_x', 1, N+1)
ref_y_param = ca.SX.sym('ref_y', 1, N+1)

# compute gripper expressions (lists -> vertcat)
x_g1_list, y_g1_list, x_g2_list, y_g2_list = [], [], [], []
for i in range(N+1):
    # x indices: 0:x1,1:y1,2:q1_1,3:q2_1,4:q3_1, 5:x2,6:y2,7:q1_2,8:q2_2,9:q3_2
    x_g1_list.append(x[0, i] + x[3, i]*ca.cos(x[2, i]))
    y_g1_list.append(x[1, i] + x[3, i]*ca.sin(x[2, i]))
    x_g2_list.append(x[5, i] + x[8, i]*ca.cos(x[7, i]))
    y_g2_list.append(x[6, i] + x[8, i]*ca.sin(x[7, i]))

A_x = ca.vertcat(*x_g1_list)  # shape (N+1, 1)
A_y = ca.vertcat(*y_g1_list)
B_x = ca.vertcat(*x_g2_list)
B_y = ca.vertcat(*y_g2_list)

one  = ca.SX(1)
zero = ca.SX(0)

# Build constraints
g = []

# Initial state equality (state at time 0 must equal provided x_original param)
g.append(x[:, 0] - x_original)             # vector of length nx

# initial object equality
g.append(object_x[:, 0] - object_x_0)     # scalar
g.append(object_y[:, 0] - object_y_0)     # scalar

# Build object_next via list of expressions (avoid in-place SX mutation)
obj_next_x_list = [object_x[:, 0]]
obj_next_y_list = [object_y[:, 0]]

# Constraints for each time step
for k in range(N):
    # 1) dynamics
    x_next = x[:, k] + B @ u[:, k]
    g.append(x[:, k+1] - x_next)  # vector length nx

    # 2) compute transform from gripper positions at k -> k+1
    M = ca.vertcat(
        ca.horzcat(A_x[k],  A_y[k], one, zero),
        ca.horzcat(A_y[k], -A_x[k], zero, one),
        ca.horzcat(B_x[k],  B_y[k], one, zero),
        ca.horzcat(B_y[k], -B_x[k], zero, one)
    )
    bvec = ca.vertcat(A_x[k+1], A_y[k+1], B_x[k+1], B_y[k+1])
    PARAM_T = ca.solve(M + 1e-7*ca.SX.eye(4), bvec)   # [a, b, tx, ty]

    a = PARAM_T[0]; b = PARAM_T[1]; tx = PARAM_T[2]; ty = PARAM_T[3]

    # object at time k (symbolic expression)
    obj_x_k = obj_next_x_list[k]
    obj_y_k = obj_next_y_list[k]

    # Standard rotation matrix representation (a=cosθ, b=sinθ)
    # p_{k+1} = R p_k + t  where R = [[a, -b],[b, a]]
    obj_x_k1 = a*obj_x_k - b*obj_y_k + tx
    obj_y_k1 = b*obj_x_k + a*obj_y_k + ty

    # append computed symbolic next
    obj_next_x_list.append(ca.reshape(obj_x_k1, 1, 1))
    obj_next_y_list.append(ca.reshape(obj_y_k1, 1, 1))

    # enforce equality with declared decision variables object_x, object_y
    g.append(object_x[:, k+1] - obj_next_x_list[k+1])
    g.append(object_y[:, k+1] - obj_next_y_list[k+1])

    # 3) distance constraint: dist_sq - d^2 >= 0  -> we encode residual and set lower bound 0 later
    dist_sq = (x[0, k] - x[5, k])**2 + (x[1, k] - x[6, k])**2
    g.append(dist_sq - d**2)

    # 4) enforce q3_1 and q3_2 to zero at each step (as constraints)
    g.append(x[4, k+1])   # q3_1 == 0
    g.append(x[9, k+1])   # q3_2 == 0

    # A_X, A_Y, B_X, B_Y updated automatically in loop
    # pack object_next lists into vertcat expressions
g.append(A_x[0] - (x[0, 0] + x[3, 0]*ca.cos(x[2, 0])))
g.append(A_y[0] - (x[1, 0] + x[3, 0]*ca.sin(x[2, 0])))
g.append(B_x[0] - (x[5, 0] + x[8, 0]*ca.cos(x[7, 0])))
g.append(B_y[0] - (x[6, 0] + x[8, 0]*ca.sin(x[7, 0])))

# Objective
cost = 0
for k in range(N):
    err = ca.vertcat(object_x[0, k+1] - ref_x_param[0, k+1],
                     object_y[0, k+1] - ref_y_param[0, k+1])
    cost += ca.mtimes([err.T, P, err])
    cost += ca.mtimes([u[:, k].T, R_ca, u[:, k]])
    dist_err = ca.vertcat((x[0, k]-x[5, k]), (x[1, k]-x[6, k]))
    cost += ca.mtimes([dist_err.T, Q * ca.DM.eye(2), dist_err])

# Decision variables vector
dec_vars = ca.vertcat(
    ca.reshape(x, -1, 1),
    ca.reshape(u, -1, 1),
    ca.reshape(object_x, -1, 1),
    ca.reshape(object_y, -1, 1)
)

g_vec = ca.vertcat(*g)

# pack parameters (flatten x_original to column)
ref_param_x = ca.reshape(ref_x_param, -1, 1)
ref_param_y = ca.reshape(ref_y_param, -1, 1)
x_original_vec = ca.reshape(x_original, -1, 1)

nlp = {
    'f': cost,
    'x': dec_vars,
    'g': g_vec,
    'p': ca.vertcat(x_original_vec, object_x_0, object_y_0, ref_param_x, ref_param_y)
}

opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# -------------------------
# Bounds
# -------------------------
x_min = np.array([-100.0, -100.0, -np.pi, 0.1, 0.0, -100.0, -100.0, -np.pi, 0.1, 0.0])
x_max = np.array([ 100.0,  100.0,  np.pi - 1e-9, 2.0, np.pi, 100.0, 100.0, np.pi - 1e-9, 2.0, np.pi])

u_min = np.array([-100.0, -100.0, -np.pi/2, -0.5, -np.pi, -100.0, -100.0, -np.pi/2, -0.5, -np.pi])
u_max = np.array([ 100.0,  100.0,  np.pi/2,  0.5,  np.pi,  100.0,  100.0,  np.pi/2,  0.5,  np.pi])

object_x_min = np.array([-100.0])
object_x_max = np.array([ 100.0])
object_y_min = np.array([-100.0])
object_y_max = np.array([ 100.0])

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

# -------------------------
# Constraint bounds (lbg, ubg)
# Must match the order and sizes of g entries above
# Order used: initial x equality (nx), object_x0 eq (1), object_y0 eq (1),
# then for each k: dynamics (nx), object_x[k+1] eq (1), object_y[k+1] eq (1),
# distance residual (1), q3_1 (1), q3_2 (1)
# -------------------------
lbg = []
ubg = []

# initial x equality
lbg += [0.0]*nx
ubg += [0.0]*nx

# initial object equalities
lbg += [0.0]; ubg += [0.0]
lbg += [0.0]; ubg += [0.0]

for k in range(N):
    # dynamics residual (nx)
    lbg += [0.0]*nx
    ubg += [0.0]*nx

    # object equalities
    lbg += [0.0]; ubg += [0.0]
    lbg += [0.0]; ubg += [0.0]

    # distance residual: dist_sq - d^2 >= 0  -> lower bound 0
    lbg += [0.0]
    ubg += [1e6]  # big upper bound

    # q3 == 0
    lbg += [0.0]; ubg += [0.0]
    lbg += [0.0]; ubg += [0.0]

lbg += [0.0]; ubg += [0.0]
lbg += [0.0]; ubg += [0.0]
lbg += [0.0]; ubg += [0.0]
lbg += [0.0]; ubg += [0.0]

# check lengths
assert len(lbg) == g_vec.shape[0], f"lbg len {len(lbg)} != g size {g_vec.shape[0]}"
assert len(ubg) == g_vec.shape[0], f"ubg len {len(ubg)} != g size {g_vec.shape[0]}"

# -------------------------
# Simulation loop
# -------------------------
# initial state and object
x0 = np.array([x1_original, y1_original, q1_1_original, q2_1_original, q3_1_original,
               x2_original, y2_original, q1_2_original, q2_2_original, q3_2_original])
x_current = x0.copy()
object_x_current = np.array([object_COM_x])
object_y_current = np.array([object_COM_y])

trajectory = [x_current.copy()]
trajectory_objx = [object_x_current[0]]
trajectory_objy = [object_y_current[0]]
controls = []

# precompute sizes & indices for extracting solver outputs
nx_block = nx * (N+1)
nu_block = nu * N
obj_block = (N+1)
total_dec_vars = nx_block + nu_block + obj_block + obj_block

# sanity check
assert total_dec_vars == (nx*(N+1) + nu*N + 2*(N+1))

for t in range(T - N):
    # prepare reference horizon
    ref_horizon_x = x_ref[t:t+N+1]
    ref_horizon_y = y_ref[t:t+N+1]

    # initial guess (warm start) - zero or previous warmstart (simple zero here)
    init_guess = np.zeros(total_dec_vars)

    # pack parameters vector p
    # x_original is expected as column vector of length nx in param order
    p_vec = np.concatenate([x_current.reshape(-1), np.array([object_x_current[0]]),
                            np.array([object_y_current[0]]),
                            np.array(ref_horizon_x).reshape(-1),
                            np.array(ref_horizon_y).reshape(-1)])

    # solve
    try:
        sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_vec)
    except RuntimeError as e:
        print(f"Solver failed at step {t} with error: {e}")
        break

    sol_x = sol['x'].full().flatten()
    print(sol_x)
    init_guess = sol_x.copy()
    # extract blocks by named indices
    idx_x = 0
    idx_u = nx_block
    idx_objx = idx_u + nu_block
    idx_objy = idx_objx + obj_block

    # first control in horizon
    u_opt = sol_x[idx_u: idx_u + nu].copy()

    # apply first control (simple Euler integrator)
    x_current = x_current + dt * u_opt

    # extract first object entries of horizon (we'll treat the 0-th as current predicted)
    objx_horizon = sol_x[idx_objx : idx_objx + obj_block]
    objy_horizon = sol_x[idx_objy : idx_objy + obj_block]

    object_x_current = np.array([objx_horizon[1]])
    object_y_current = np.array([objy_horizon[1]])

    # store
    trajectory.append(x_current.copy())
    trajectory_objx.append(object_x_current[0])
    trajectory_objy.append(object_y_current[0])
    controls.append(u_opt.copy())

    print(f"step {t}: first-control u[0]={u_opt[0]:.4f}, objx={object_x_current[0]:.4f}, objy={object_y_current[0]:.4f}")

# convert to arrays for plotting
trajectory = np.array(trajectory)            # shape (steps, nx)
trajectory_objx = np.array(trajectory_objx)
trajectory_objy = np.array(trajectory_objy)

# Plotting
plt.figure(figsize=(8,6))
sim_time = np.arange(0, len(trajectory_objx)) * dt
plt.plot(t_grid[0:len(trajectory_objx)], y_ref[0:len(trajectory_objx)], 'r--', label='Reference y (tanh)')
# plt.plot(sim_time, trajectory_objy, 'b-', label='Object Y (sim)')
# plt.plot(sim_time, trajectory_objx, 'm-', label='Object X (sim)')
plt.plot(trajectory_objx, trajectory_objy, 'k-', label='Object XY (path)')
plt.plot(trajectory[:,0], trajectory[:,1], 'g-', label='Bot1 XY (path)')
plt.plot(trajectory[:,5], trajectory[:,6], 'y-', label='Bot2 XY (path)')
plt.xlabel('Time or X')
plt.ylabel('Position')
plt.legend()
plt.title('MPC simulation: robot paths and object trajectory')
plt.grid(True)
plt.show()
