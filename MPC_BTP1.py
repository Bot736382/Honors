# mpc_fixed_with_tests.py
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math
import sys

# -------------------------
# Unit tests (transform & distance constraint)
# -------------------------
plot_count = 1
animation_count = 1
def solve_transform_numpy(Ax_k, Ay_k, Bx_k, By_k, Ax_k1, Ay_k1, Bx_k1, By_k1, reg=0):
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

def animate_robots(trajectory, trajectory_objx, trajectory_objy, x_ref, y_ref, dt, save_path="robot_animation.mp4"):
    """
    Animate two robots as triangles with centroid and heading direction.
    trajectory: (T, nx) array where columns are [x1, y1, q1_1, q2_1, q3_1, x2, y2, q1_2, q2_2, q3_2]
    trajectory_objx, trajectory_objy: object center trajectory
    x_ref, y_ref: reference trajectory
    dt: time step
    save_path: output video file
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # triangle size (half-width and height for visualization)
    triangle_size = 0.25
    
    def get_triangle(cx, cy, heading, tri_size):
        # triangle points in local frame (before rotation)
        local_pts = np.array([
            [tri_size, 0],        # front corner (tip)
            [-tri_size, tri_size],  # back-left
            [-tri_size, -tri_size]  # back-right
        ])
        # rotation matrix
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading), np.cos(heading)]])
        # rotate and translate
        rotated = R @ local_pts.T  
        vertices = rotated.T + np.array([cx, cy])
        return vertices
    
    def init_anim():
        ax.clear()
        return []
    
    def animate(frame):
        ax.clear()
        
        # extract positions from trajectory
        x1, y1, q1_1, q2_1, q3_1 = trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 2], trajectory[frame, 3], trajectory[frame, 4]
        x2, y2, q1_2, q2_2, q3_2 = trajectory[frame, 5], trajectory[frame, 6], trajectory[frame, 7], trajectory[frame, 8], trajectory[frame, 9]
        
        # object position
        obj_x = trajectory_objx[frame]
        obj_y = trajectory_objy[frame]
        
        # compute gripper positions
        x1g = x1 + q2_1 * np.cos(q1_1)
        y1g = y1 + q2_1 * np.sin(q1_1)
        x2g = x2 + q2_2 * np.cos(q1_2)
        y2g = y2 + q2_2 * np.sin(q1_2)
        
        # plot reference trajectory
        ax.plot(x_ref[0:len(trajectory_objx)], y_ref[0:len(trajectory_objx)], 'r--', linewidth=2, label='Reference path')
        
        # plot object trajectory up to current frame
        ax.plot(trajectory_objx[0:frame+1], trajectory_objy[0:frame+1], 'k-', linewidth=1.5, label='Object trajectory')
        
        # plot bot1 trajectory up to current frame
        ax.plot(trajectory[0:frame+1, 0], trajectory[0:frame+1, 1], 'g-', linewidth=1, alpha=0.6, label='Bot1 trajectory')
        
        # plot bot2 trajectory up to current frame
        ax.plot(trajectory[0:frame+1, 5], trajectory[0:frame+1, 6], 'b-', linewidth=1, alpha=0.6, label='Bot2 trajectory')
        
        # draw bot1 as triangle with heading
        tri1 = get_triangle(x1, y1, q1_1, triangle_size)
        tri1_patch = plt.Polygon(tri1, fill=True, edgecolor='green', facecolor='lightgreen', linewidth=2, alpha=0.7)
        ax.add_patch(tri1_patch)
        
        # draw heading line for bot1 (from centroid to front corner)
        front1 = tri1[0]
        ax.plot([x1, front1[0]], [y1, front1[1]], 'g-', linewidth=2, label='Bot1 heading')
        
        # draw line from bot1 centroid to gripper
        ax.plot([x1, x1g], [y1, y1g], 'g--', linewidth=3.5, alpha=0.8, label='Bot1 gripper arm')
        
        # draw bot2 as triangle with heading
        tri2 = get_triangle(x2, y2, q1_2, triangle_size)
        tri2_patch = plt.Polygon(tri2, fill=True, edgecolor='blue', facecolor='lightblue', linewidth=2, alpha=0.7)
        ax.add_patch(tri2_patch)
        
        # draw heading line for bot2
        front2 = tri2[0]
        ax.plot([x2, front2[0]], [y2, front2[1]], 'b-', linewidth=2, label='Bot2 heading')
        
        # draw line from bot2 centroid to gripper
        ax.plot([x2, x2g], [y2, y2g], 'b--', linewidth=3.5, alpha=0.8, label='Bot2 gripper arm')
        
        # plot object center
        ax.plot(obj_x, obj_y, 'ko', markersize=10, label='Object center')
        
        # gripper positions (markers)
        ax.plot(x1g, y1g, 'g^', markersize=6, label='Bot1 gripper')
        ax.plot(x2g, y2g, 'b^', markersize=6, label='Bot2 gripper')
        
        # set axis properties
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'Robot Animation - Step {frame} / {len(trajectory)-1} (time={frame*dt:.2f}s)')
        
        return ax.patches + ax.lines
    
    # create animation
    anim = ani.FuncAnimation(fig, animate, init_func=init_anim, frames=len(trajectory), 
                            interval=100, blit=False, repeat=True)
    
    # save animation
    # save animation
    gif_path = save_path.replace('.mp4', '.gif')
    try:
        anim.save(save_path, writer='ffmpeg', fps=10, dpi=100)
        print(f"✓ MP4 animation saved to {save_path}")
    except Exception as e:
        print(f"⚠ ffmpeg not available: {e}")
        print(f"Saving as GIF instead to {gif_path}...")
        try:
            # fallback to PIL (pillow) — supports .gif
            anim.save(gif_path, writer='pillow', fps=10)
            print(f"✓ GIF animation saved to {gif_path}")
        except Exception as e2:
            print(f"GIF save also failed: {e2}")
            print("Showing animation live instead...")
            plt.show()
    
    plt.close()

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
    R = np.array([[a_true, b_true],[-b_true, a_true]])
    t = np.array([tx_true, ty_true])
    A_k1 = (R @ A_k) + t
    B_k1 = (R @ B_k) + t

    sol = solve_transform_numpy(A_k[0], A_k[1], B_k[0], B_k[1], A_k1[0], A_k1[1], B_k1[0], B_k1[1])
    a_est, b_est, tx_est, ty_est = sol

    ok = np.allclose([a_est,b_est,tx_est,ty_est], [a_true,b_true,tx_true,ty_true], atol=1e-3)
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
    print(ok1)
    print(ok2)
    return ok1 and ok2

if __name__ == "__main__":
    # run unit tests first
    t1 = unit_test_transform()
    t2 = unit_test_distance()
    if not (t1 and t2):
        print("Unit tests failed - stop. Fix unit test failures first.")
        sys.exit(1)
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
q1_1_original = 7*np.pi/4
q2_1_original = 0.2
q3_1_original = 3*np.pi/4

x2_original = 0.0
y2_original = 1.0
q1_2_original = 3*np.pi/4
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
l1 = math.sqrt((object_COM_x-x1g_original)**2 + (object_COM_y-y1g_original)**2)
l2 = math.sqrt((object_COM_x-x2g_original)**2 + (object_COM_y-y2g_original)**2)
# Hyperparameters
dt = 0.1
N = 10              # MPC horizon
print(f"N: {N}")
T = 100           # total simulation steps

nu = 10 # dimension of u per step (same as nx here)
nx = 10 # dimension of state

B = dt * np.eye(nx)

t_grid = np.linspace(0, T*dt, T+1)
x_ref = np.linspace(0, T*dt, T+1)          # example x reference (unused beyond demo)
y_ref =2*np.cos(x_ref)
# y_ref =4*np.sin(t_grid)                     # example y reference

P = np.array([[20.0, 0.0],[0.0, 20.0]])
Q = 10.0
# R = 0.1 * np.eye(nu)
# R = np.array([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10])
R = np.array([1.0, 1.0, 0.1, 1, 0.1, 1.0, 1, 0.1, 1, 0.1])
diag_R = 5*np.diag(R)
# print(diag_R)
R_ca = ca.SX(diag_R)

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
    PARAM_T = ca.solve(M + 1e-9*ca.SX.eye(4), bvec)   # [a, b, tx, ty]

    a = PARAM_T[0]; b = PARAM_T[1]; tx = PARAM_T[2]; ty = PARAM_T[3]

    # object at time k (symbolic expression)
    obj_x_k = obj_next_x_list[k]
    obj_y_k = obj_next_y_list[k]

    # Standard rotation matrix representation (a=cosθ, b=sinθ)
    ### ERROR
    # p_{k+1} = R p_k + t  where R = [[a, b],[-b, a]]
    obj_x_k1 = a*obj_x_k + b*obj_y_k + tx
    obj_y_k1 = -b*obj_x_k + a*obj_y_k + ty

    # append computed symbolic next
    obj_next_x_list.append(ca.reshape(obj_x_k1, 1, 1))
    obj_next_y_list.append(ca.reshape(obj_y_k1, 1, 1))

    # enforce equality with declared decision variables object_x, object_y
    g.append(object_x[:, k+1] - obj_next_x_list[k+1])
    g.append(object_y[:, k+1] - obj_next_y_list[k+1])

    # 3) distance constraint: dist_sq - d^2 >= 0  
    dist_sq = (x[0, k] - x[5, k])**2 + (x[1, k] - x[6, k])**2
    g.append(dist_sq - d**2)

    # 4) enforce q3_1 and q3_2 to zero at each step (as constraints)
    g.append(x[4, k+1])   # q3_1 == 0
    g.append(x[9, k+1])   # q3_2 == 0

    # A_X, A_Y, B_X, B_Y updated automatically in loop
    # pack object_next lists into vertcat expressions
    g.append(A_x[k] - (x[0, k] + x[3, k]*ca.cos(x[2, k])))
    g.append(A_y[k] - (x[1, k] + x[3, k]*ca.sin(x[2, k])))
    g.append(B_x[k] - (x[5, k] + x[8, k]*ca.cos(x[7, k])))
    g.append(B_y[k] - (x[6, k] + x[8, k]*ca.sin(x[7, k])))

    # # Distance between grippers and object must be equal to l1, l2
    # dist1_sq = (object_x[:, k] - (x[0, k] + x[3, k]*ca.cos(x[2, k])))**2 + (object_y[:, k] - (x[1, k] + x[3, k]*ca.sin(x[2, k])))**2
    # g.append(dist1_sq - l1**2)
    dist2_sq = (object_x[:, k] - (x[5, k] + x[8, k]*ca.cos(x[7, k])))**2 + (object_y[:, k] - (x[6, k] + x[8, k]*ca.sin(x[7, k])))**2
    g.append(dist2_sq - l2**2)
    

# Objective
cost = 0
for k in range(N):
    err = ca.vertcat(object_x[0, k] - ref_x_param[0, k],
                     object_y[0, k] - ref_y_param[0, k])
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

    # # distance constraints for grippers to object
    # lbg += [0.0]; ubg += [0.0]
    # lbg += [0.0]; ubg += [0.0]

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
trajectory = np.array(trajectory)           
trajectory_objx = np.array(trajectory_objx)
trajectory_objy = np.array(trajectory_objy)
controls_final = np.array(controls)

# write to csv
np.savetxt("mpc_trajectory.csv", trajectory, delimiter=",")
np.savetxt("mpc_object_trajectory.csv", np.vstack([trajectory_objx, trajectory_objy]).T, delimiter=",")


# # Plotting the trajectories and controls in 2 subplots
# plt.figure(figsize=(8,6))
# sim_time = np.arange(0, len(trajectory_objx)) * dt
# # plt.plot(t_grid[0:len(trajectory_objx)], y_ref[0:len(trajectory_objx)], 'r--', label='Reference y (tanh)')
# plt.plot(x_ref[0:len(trajectory_objx)], y_ref[0:len(trajectory_objx)], 'r--', label='Reference path (x,y)')
# # plt.plot(sim_time, trajectory_objy, 'b-', label='Object Y (sim)')
# # plt.plot(sim_time, trajectory_objx, 'm-', label='Object X (sim)')
# plt.plot(trajectory_objx, trajectory_objy, 'k-', label='Object XY (path)')
# # plt.plot(trajectory[:,0]+ trajectory[:,3]*np.cos(trajectory[:,2]), trajectory[:,1]+ trajectory[:,3]*np.sin(trajectory[:,2]), 'b-', label='A1 XY (sim)')
# # plt.plot(trajectory[:,5]+ trajectory[:,8]*np.cos(trajectory[:,7]), trajectory[:,6]+ trajectory[:,8]*np.sin(trajectory[:,7]), 'm-', label='B1 XY (sim)')
# plt.plot(trajectory[:,0], trajectory[:,1], 'g-', label='Bot1 XY (path)')
# plt.plot(trajectory[:,5], trajectory[:,6], 'y-', label='Bot2 XY (path)')
# # plt.plot(sim_time, trajectory[:,3], 'c--', label='A1 gripper length')
# # plt.plot(sim_time, trajectory[:,8], 'r--', label='B1 gripper length')
# plt.xlabel('Time or X')
# plt.ylabel('Position')
# plt.legend()
# plt.title('MPC simulation: robot paths and object trajectory')
# plt.savefig("MPC_simulation_R_non_0.png")
# plt.grid(True)

# plt.figure(figsize=(8,6))
# # plt.subplot(2,1,2)
# plt.plot(sim_time[0:len(controls_final)], controls_final[:,0], 'b-', label='x-vel Bot1')
# plt.plot(sim_time[0:len(controls_final)], controls_final[:,5], 'r-', label='x-vel Bot2')
# plt.plot(sim_time[0:len(controls_final)], controls_final[:,1], 'b--', label='y-vel Bot1')
# plt.plot(sim_time[0:len(controls_final)], controls_final[:,6], 'r--', label='y-vel Bot2')
# plt.xlabel('Time')
# plt.ylabel('Velocity controls')
# plt.legend()
# plt.title('MPC simulation: control inputs over time')
# plt.savefig("Controls_with_R_non_0.png")
# plt.grid(True)
# plt.show()
if (plot_count == 1):
    fig, axs = plt.subplots(2, 1, figsize=(8,10))

    sim_time = np.arange(0, len(trajectory_objx)) * dt

    ###########################################
    # Subplot 1 — Paths
    ###########################################
    ax = axs[0]

    ax.plot(x_ref[0:len(trajectory)], y_ref[0:len(trajectory)], 
            'r--', label='Reference path (x,y)')

    ax.plot(trajectory_objx[0:len(trajectory)], trajectory_objy[0:len(trajectory)], 
            'k-', label='Object XY (path)')

    ax.plot(trajectory[:,0], trajectory[:,1], 
            'g-', label='Bot1 (path)')

    ax.plot(trajectory[:,5], trajectory[:,6], 
            'y-', label='Bot2 (path)')

    ax.plot(trajectory[:,0] + trajectory[:,3]*np.cos(trajectory[:,2]), 
            trajectory[:,1] + trajectory[:,3]*np.sin(trajectory[:,2]),
            'g--', label='Bot1 gripper path')
    ax.plot(trajectory[:,5] + trajectory[:,8]*np.cos(trajectory[:,7]),
            trajectory[:,6] + trajectory[:,8]*np.sin(trajectory[:,7]),
            'y--', label='Bot2 gripper path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('MPC simulation: robot paths and object trajectory')

    # Major + minor grid
    ax.grid(True, which='major', linewidth=0.8)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.minorticks_on()

    ax.legend()


    ###########################################
    # Subplot 2 — Controls
    ###########################################
    ax = axs[1]

    ax.plot(sim_time[0:len(controls_final)], controls_final[:,0], 
            'b-', label='x-vel Bot1')
    ax.plot(sim_time[0:len(controls_final)], controls_final[:,5], 
            'r-', label='x-vel Bot2')
    ax.plot(sim_time[0:len(controls_final)], controls_final[:,1], 
            'b--', label='y-vel Bot1')
    ax.plot(sim_time[0:len(controls_final)], controls_final[:,6], 
            'r--', label='y-vel Bot2')

    ax.set_xlabel('Time')
    ax.set_ylabel('Velocity controls')
    ax.set_title('MPC simulation: control inputs over time')

    # Major + minor grid
    ax.grid(True, which='major', linewidth=0.8)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.4, alpha=0.7)
    ax.minorticks_on()

    ax.legend()

    plt.tight_layout()
    plt.savefig("MPC_sim_and_controls_with_minor_grids.png")
    plt.show()

# -------------------------
# Animation of robot trajectories
# -------------------------

# Run animation
if (animation_count==1):
    print("Generating animation...")
    animate_robots(trajectory, trajectory_objx, trajectory_objy, x_ref, y_ref, dt, save_path="robot_animation.mp4")

for i in range(len(trajectory_objx)):
    # print distance between grippers and object
    dist = np.sqrt((trajectory_objx[i] - trajectory[i, 0])**2 + (trajectory_objy[i] - trajectory[i, 1])**2)
    # print distance between the two bots
    dist_bots = np.sqrt((trajectory[i, 0] - trajectory[i, 5])**2 + (trajectory[i, 1] - trajectory[i, 6])**2)
    print(f"Step {i}: distance between the bots: {dist_bots:.4f}")

