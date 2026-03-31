"""
NMPC - 5 ROBOTS PENTAGON FORMATION
===================================
5 robots carrying a pentagonal object with velocity-based CBF
Static and dynamic obstacle avoidance

Robots are positioned at the midpoints of pentagon edges,
holding a rigid pentagonal object.
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Polygon
import time
import sys

# ==================== SAT Collision Detection ====================
def sat_gap_fixed(vertices1, vertices2, eps=1e-4):  
    """Separating Axis Theorem for polygon collision detection"""
    def get_normals(verts):
        v_next = ca.horzcat(verts[:, 1:], verts[:, 0:1])
        edges = v_next - verts
        normals = ca.vertcat(-edges[1, :], edges[0, :])
        lengths = ca.sqrt(ca.sum1(edges**2) + eps)
        normals = normals / ca.repmat(lengths, 2, 1)
        return normals
    
    normals1 = get_normals(vertices1)
    normals2 = get_normals(vertices2)
    all_normals = ca.horzcat(normals1, normals2)
    
    proj1 = ca.mtimes(all_normals.T, vertices1)
    proj2 = ca.mtimes(all_normals.T, vertices2)
    
    min1 = ca.DM()
    max1 = ca.DM()
    for i in range(proj1.size()[0]):
        min1 = ca.horzcat(min1, ca.mmin(proj1[i, :]))
        max1 = ca.horzcat(max1, ca.mmax(proj1[i, :]))

    min2 = ca.DM()
    max2 = ca.DM()
    for i in range(proj2.size()[0]):
        min2 = ca.horzcat(min2, ca.mmin(proj2[i, :]))
        max2 = ca.horzcat(max2, ca.mmax(proj2[i, :]))

    gaps = ca.fmin(max1, max2) - ca.fmax(min1, min2)
    return ca.mmin(gaps)


def create_hexagon_symbolic(center, radius):
    """Create hexagon vertices as symbolic CasADi expression"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = []
    for angle in angles:
        x = center[0] + radius * ca.cos(angle)
        y = center[1] + radius * ca.sin(angle)
        vertices.append(ca.vertcat(x, y))
    return ca.horzcat(*vertices)


def create_hexagon_numpy(center, radius):
    """Create hexagon vertices as numpy array"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = np.zeros((2, 6))
    for i, angle in enumerate(angles):
        vertices[0, i] = center[0] + radius * np.cos(angle)
        vertices[1, i] = center[1] + radius * np.sin(angle)
    return vertices


# ==================== TRIANGLE POINT GENERATION ====================
def calculate_triangle_points_sym(A, R, r):
    """Calculate triangle points for robot representation"""
    vec_RA = A - R
    d_RA = ca.norm_2(vec_RA) + 1e-6
    theta = ca.asin(ca.fmin(r / d_RA, 0.999))
    u_RA = vec_RA / d_RA
    vec_RF = (d_RA - r * ca.cos(theta)) * u_RA
    F = R + vec_RF
    d_perp = r * ca.sin(theta)
    u_perp = ca.vertcat(-u_RA[1], u_RA[0])
    B = F + d_perp * u_perp
    C = F - d_perp * u_perp
    return A, B, C, theta


# ==================== ANGULAR NON-COLLINEARITY CONSTRAINT ====================
def compute_proper_angular_constraint(robot_base, ee_pos, vertex_left, vertex_right, 
                                      robot_radius, theta_min_deg=15.0):
    """PROPER CONSTRAINT for angular non-collinearity"""
    A, B, C, theta_triangle = calculate_triangle_points_sym(ee_pos, robot_base, robot_radius)
    
    edge_left = vertex_left - ee_pos
    edge_right = vertex_right - ee_pos
    
    edge_left_norm = ca.norm_2(edge_left) + 1e-6
    edge_right_norm = ca.norm_2(edge_right) + 1e-6
    u_edge_left = edge_left / edge_left_norm
    u_edge_right = edge_right / edge_right_norm
    
    vec_AB = B - ee_pos
    vec_AB_norm = ca.norm_2(vec_AB) + 1e-6
    u_AB = vec_AB / vec_AB_norm
    
    vec_AC = C - ee_pos
    vec_AC_norm = ca.norm_2(vec_AC) + 1e-6
    u_AC = vec_AC / vec_AC_norm
    
    cos_angle_AB_left = ca.dot(u_AB, u_edge_left)
    cos_angle_AC_right = ca.dot(u_AC, u_edge_right)
    
    theta_min = theta_min_deg * np.pi / 180.0
    cos_theta_min = ca.cos(theta_min)
    
    h_AB_left = cos_theta_min - ca.fabs(cos_angle_AB_left)
    h_AC_right = cos_theta_min - ca.fabs(cos_angle_AC_right)
    
    return ca.fmin(h_AB_left, h_AC_right)


# ==================== DYNAMIC OBSTACLE TRAJECTORY GENERATORS ====================
def generate_circular_trajectory(center, radius, omega, t_vec):
    """Generate circular motion trajectory"""
    traj = np.zeros((2, len(t_vec)))
    traj[0, :] = center[0] + radius * np.cos(omega * t_vec)
    traj[1, :] = center[1] + radius * np.sin(omega * t_vec)
    return traj


def generate_linear_trajectory(start, velocity, t_vec):
    """Generate linear motion trajectory"""
    traj = np.zeros((2, len(t_vec)))
    traj[0, :] = start[0] + velocity[0] * t_vec
    traj[1, :] = start[1] + velocity[1] * t_vec
    return traj


def get_velocity_from_trajectory(traj, idx, dt):
    """Compute velocity from trajectory using finite difference"""
    if idx < len(traj[0]) - 1:
        vel = (traj[:, idx+1] - traj[:, idx]) / dt
    else:
        vel = (traj[:, idx] - traj[:, idx-1]) / dt
    return vel


# ==================== PENTAGON GEOMETRY ====================
def create_pentagon_vertices(center, radius, rotation=0):
    """Create pentagon vertices centered at `center` with circumradius `radius`"""
    vertices = []
    for i in range(5):
        angle = rotation + 2 * np.pi * i / 5 - np.pi/2  # Start from top
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append(np.array([x, y]))
    return np.array(vertices)


def get_pentagon_midpoints(vertices):
    """Get midpoints of pentagon edges (grasp points for robots)"""
    midpoints = []
    n = len(vertices)
    for i in range(n):
        midpoint = (vertices[i] + vertices[(i + 1) % n]) / 2
        midpoints.append(midpoint)
    return np.array(midpoints)


# ==================== PARAMETERS ====================
dt = 0.1
N = 20  # Increased from 8 - longer horizon to see past obstacles
time_steps = 180  # Longer trajectory

NUM_ROBOTS = 5  # Changed to 5 robots
n_arm_joints = 3
n_states_single = 2 + n_arm_joints
n_states = NUM_ROBOTS * n_states_single
n_controls_single = 2 + n_arm_joints
n_controls = NUM_ROBOTS * n_controls_single
L_arm1, L_arm3 = 0.05, 0.05
arm_link_params = [L_arm1, L_arm3]
robot_radius = 0.15

# Pentagon object parameters
pentagon_radius = 0.6  # Circumradius of the pentagon object

Q_obj = np.diag([150.0, 150.0])  # Increased from 80 - stronger push to goal
# Higher costs for arm movements to keep them stable
R_diag_single = [1.0, 1.0, 10.0, 50.0, 10.0]  # Reduced base velocity cost
R = ca.diag(np.tile(R_diag_single, NUM_ROBOTS))
Q_terminal_obj = np.diag([600.0, 600.0])  # Increased terminal cost

# Cost hierarchy
W_rigidity_slack = 10000000.0  # Extremely high weight - rigidity is paramount
cost_cbf_slack = 500000.0

# CBF parameters - VELOCITY-BASED
gamma_cbf_velocity = 10.0  # Increased for more aggressive avoidance
gamma_base_object = 12.0
d_min_inter = 0.20
theta_min_collinearity = 15.0
d_min_obstacle_base = 0.50   # Reduced from 0.60 for dynamic obstacles
d_min_obstacle_object = 0.40 # Reduced from 0.50 - tighter clearance to pass through
d_min_base_object = 0.50
max_cbf_slack = 0.01  # Increased slack allowance for tight spots

# Smart gating - reduced since we have formation constraints
GATE_INTER_ROBOT = 0.3  # Reduced - formation constraints handle most cases

# Control limits
vx_max, vy_max = 5.0, 5.0  # Increased from 4.0 for faster gap traversal
dq1_arm_max, d2_dot_arm_max, dq3_arm_max = np.pi/2, 0.15, np.pi/2
u_max_single = [vx_max, vy_max, dq1_arm_max, d2_dot_arm_max, dq3_arm_max]
u_min_single = [-vx_max, -vy_max, -dq1_arm_max, -0.15, -dq3_arm_max]
q1_arm_max, q3_arm_max = np.pi, np.pi
d2_arm_min, d2_arm_max = 0.1, 0.5
lbq_single = [-20, -20, -q1_arm_max, d2_arm_min, -q3_arm_max]
ubq_single = [20, 20, q1_arm_max, d2_arm_max, q3_arm_max]
u_min, u_max = np.tile(u_min_single, NUM_ROBOTS), np.tile(u_max_single, NUM_ROBOTS)
lbq, ubq = np.tile(lbq_single, NUM_ROBOTS), np.tile(ubq_single, NUM_ROBOTS)

# Initial conditions - Pentagon setup
start_point_offset_x, start_point_offset_y = 1.5, 2.5
x0 = np.zeros(n_states)
initial_d2_length = 0.3
initial_arm_length = L_arm1 + initial_d2_length + L_arm3 + 0.1

# Create pentagon object vertices and grasp points (midpoints)
pentagon_vertices = create_pentagon_vertices(np.array([start_point_offset_x, start_point_offset_y]), pentagon_radius)
grasp_points = get_pentagon_midpoints(pentagon_vertices)
obj_centroid_initial = np.mean(pentagon_vertices, axis=0)

print("\n" + "="*80)
print("5-ROBOT PENTAGON FORMATION SETUP")
print("="*80)
print(f"Pentagon center: ({start_point_offset_x:.2f}, {start_point_offset_y:.2f})")
print(f"Pentagon circumradius: {pentagon_radius:.2f}m")
print(f"Number of robots: {NUM_ROBOTS}")
print("Grasp points (midpoints of pentagon edges):")
for i, gp in enumerate(grasp_points):
    print(f"  Robot {i}: ({gp[0]:.3f}, {gp[1]:.3f})")
print("="*80 + "\n")

# Initialize robot states
for i in range(NUM_ROBOTS):
    ee_pos = grasp_points[i]
    vec_cent_to_ee = ee_pos - obj_centroid_initial
    unit_vec = vec_cent_to_ee / (np.linalg.norm(vec_cent_to_ee) + 1e-9)
    base_pos = ee_pos + unit_vec * initial_arm_length
    base_x, base_y = base_pos[0], base_pos[1]
    q1 = np.arctan2(ee_pos[1] - base_y, ee_pos[0] - base_x)
    d2, q3 = initial_d2_length, 0.0
    x0[i * n_states_single:(i + 1) * n_states_single] = [base_x, base_y, q1, d2, q3]

# ==================== KINEMATICS ====================
def fk_ee_world_np(base, arm, params):
    """Forward kinematics"""
    x_b, y_b = base
    q1, d2, q3 = arm
    ex = x_b + (params[0] + d2) * np.cos(q1) + params[1] * np.cos(q1 + q3)
    ey = y_b + (params[0] + d2) * np.sin(q1) + params[1] * np.sin(q1 + q3)
    return np.array([ex, ey])


def fk_ee_sym(b, a, p):
    """Symbolic forward kinematics for CasADi"""
    ex = b[0] + (p[0] + a[1]) * ca.cos(a[0]) + p[1] * ca.cos(a[0] + a[2])
    ey = b[1] + (p[0] + a[1]) * ca.sin(a[0]) + p[1] * ca.sin(a[0] + a[2])
    return ca.vertcat(ex, ey)


def calculate_polygon_centroid_np(vertices):
    return np.mean(vertices, axis=0)


initial_ees = np.array([fk_ee_world_np(x0[i*n_states_single:i*n_states_single+2], 
                                        x0[i*n_states_single+2:(i+1)*n_states_single], 
                                        arm_link_params) for i in range(NUM_ROBOTS)])
obj_centroid_0_np = calculate_polygon_centroid_np(initial_ees)


def generate_straight_line_traj_np(start_pos, end_pos, t_vec):
    """Generate a straight line trajectory"""
    X_ref = np.zeros((2, len(t_vec)))
    progress = t_vec / t_vec[-1] if t_vec[-1] > 0 else np.zeros(len(t_vec))
    X_ref[0, :] = start_pos[0] + progress * (end_pos[0] - start_pos[0])
    X_ref[1, :] = start_pos[1] + progress * (end_pos[1] - start_pos[1])
    return X_ref

def generate_sinusoidal_traj_np(start_pos, amplitude, frequency, t_vec):
    """Generate a sinusoidal trajectory in y-direction"""
    X_ref = np.zeros((2, len(t_vec)))
    X_ref[0, :] = start_pos[0] + 0.5 * t_vec  # Constant forward speed
    X_ref[1, :] = start_pos[1] + amplitude * np.sin(frequency * t_vec)
    return X_ref

# Reference trajectory for object - STRAIGHT LINE for clean motion
obj_start = obj_centroid_0_np
obj_end = np.array([14.0, 2.5])
t_vec = np.arange(0, dt * time_steps, dt)
#X_ref_obj_full = generate_straight_line_traj_np(obj_start, obj_end, t_vec)
X_ref_obj_full = generate_sinusoidal_traj_np(obj_start, amplitude=1.0, frequency=0.2, t_vec=t_vec)

# ==================== STATIC OBSTACLES ====================
num_static_obstacles = 3
static_obstacle_radius = 0.35  # Reduced from 0.4
static_obstacle_centers = [
    np.array([4.0, 3.5]),   # Moved up and left
    np.array([12.0, 1.5]),
    np.array([7,7.5])   # Moved right and down
]

print("\n" + "="*80)
print("OBSTACLE CONFIGURATION")
print("="*80)
print(f"STATIC OBSTACLES: {num_static_obstacles}")
for i, center in enumerate(static_obstacle_centers):
    print(f"  Static {i+1}: position = {center}, radius = {static_obstacle_radius}m")

# ==================== DYNAMIC OBSTACLES ====================
num_dynamic_obstacles = 2
dynamic_obstacle_radius = 0.30  # Reduced from 0.35

# Dynamic obstacle 1: Circular motion - moved lower
dyn_obs_1_center = np.array([8.0, 1.5])  # Moved down from 2.5
dyn_obs_1_orbit_radius = 0.8  # Reduced from 1.0
dyn_obs_1_omega = 0.3  # Slightly faster
dyn_obs_1_traj = generate_circular_trajectory(dyn_obs_1_center, dyn_obs_1_orbit_radius, 
                                               dyn_obs_1_omega, t_vec)

# Dynamic obstacle 2: Oscillating motion - moved higher
dyn_obs_2_start = np.array([7.5, 4.2])  # Moved up from 4.0
dyn_obs_2_traj = np.zeros((2, len(t_vec)))
for i, t in enumerate(t_vec):
    x_offset = 2.0 * np.sin(0.2 * t)  # Reduced amplitude from 2.5
    dyn_obs_2_traj[0, i] = dyn_obs_2_start[0] + x_offset
    dyn_obs_2_traj[1, i] = dyn_obs_2_start[1]

print(f"\nDYNAMIC OBSTACLES: {num_dynamic_obstacles}")
print(f"  Dynamic 1: circular motion around {dyn_obs_1_center}, radius = {dyn_obs_1_orbit_radius}m")
print(f"  Dynamic 2: oscillating motion from {dyn_obs_2_start}")
print("  NOTE: Obstacles spread apart vertically for better passage")
print("="*80 + "\n")

all_obstacle_trajs = [dyn_obs_1_traj, dyn_obs_2_traj]

# ==================== OPTIMIZATION SETUP ====================
opti = ca.Opti()
X = opti.variable(n_states, N + 1)
U = opti.variable(n_controls, N)
X_ref_p = opti.parameter(2, N + 1)
x0_p = opti.parameter(n_states)

# STATIC obstacle parameters
static_obs_center_p = [opti.parameter(2) for _ in range(num_static_obstacles)]

# DYNAMIC obstacle parameters (position AND velocity)
dynamic_obs_center_p = [opti.parameter(2) for _ in range(num_dynamic_obstacles)]
dynamic_obs_vel_p = [opti.parameter(2) for _ in range(num_dynamic_obstacles)]

# CBF slack variables - UPDATED COUNT FOR 5 ROBOTS
n_inter_robot_pairs = NUM_ROBOTS * (NUM_ROBOTS - 1) // 2  # 10 pairs for 5 robots
n_robot_base_static_obs = NUM_ROBOTS * num_static_obstacles
n_ee_static_obs = NUM_ROBOTS * num_static_obstacles
n_robot_base_dynamic_obs = NUM_ROBOTS * num_dynamic_obstacles
n_ee_dynamic_obs = NUM_ROBOTS * num_dynamic_obstacles
# REMOVED: n_base_object - this was causing robots to swing outward
n_angular = NUM_ROBOTS

n_cbf_constraints = (n_inter_robot_pairs + n_robot_base_static_obs + n_ee_static_obs + 
                     n_robot_base_dynamic_obs + n_ee_dynamic_obs + n_angular)

print(f"Total CBF constraints: {n_cbf_constraints}")
print(f"  - Inter-robot: {n_inter_robot_pairs}")
print(f"  - Base to static obstacles: {n_robot_base_static_obs}")
print(f"  - EE to static obstacles: {n_ee_static_obs}")
print(f"  - Base to dynamic obstacles: {n_robot_base_dynamic_obs}")
print(f"  - EE to dynamic obstacles: {n_ee_dynamic_obs}")
print(f"  - Angular: {n_angular}")
print(f"  NOTE: Removed base-to-object CBF (was causing outward swings)\n")
# sys.exit()
cbf_slack = opti.variable(n_cbf_constraints, N)
opti.subject_to(opti.bounded(0, cbf_slack, max_cbf_slack))

# Rigidity slack - 10 pairs for pentagon (all pairwise distances)
n_rigidity = NUM_ROBOTS * (NUM_ROBOTS - 1) // 2  # 10 for 5 robots
slack_rigidity = opti.variable(n_rigidity, N)
max_rigidity_slack = 0.001  # Very small slack allowed
opti.subject_to(opti.bounded(0, slack_rigidity, max_rigidity_slack))

# Initial condition
opti.subject_to(X[:, 0] == x0_p)

# Dynamics
for k in range(N):
    for i in range(NUM_ROBOTS):
        idx_base = i * n_states_single
        idx_arm = idx_base + 2
        base_k = X[idx_base:idx_base+2, k]
        arm_k = X[idx_arm:idx_arm+n_arm_joints, k]
        u_k = U[i*n_controls_single:(i+1)*n_controls_single, k]
        
        base_next = base_k + dt * u_k[:2]
        arm_next = arm_k + dt * u_k[2:]
        
        opti.subject_to(X[idx_base:idx_base+2, k+1] == base_next)
        opti.subject_to(X[idx_arm:idx_arm+n_arm_joints, k+1] == arm_next)

# State bounds
for k in range(N + 1):
    for i in range(NUM_ROBOTS):
        idx = i * n_states_single
        opti.subject_to(opti.bounded(lbq_single, X[idx:idx+n_states_single, k], ubq_single))

# Control bounds
for k in range(N):
    opti.subject_to(opti.bounded(u_min, U[:, k], u_max))

# ==================== COST FUNCTION ====================
cost = 0

# Compute initial offsets from centroid for each robot base
initial_base_offsets = []
for i in range(NUM_ROBOTS):
    base_i = x0[i*n_states_single:i*n_states_single+2]
    offset = base_i - obj_centroid_0_np
    initial_base_offsets.append(offset)

# Weight for formation keeping - reduced to allow flexibility in tight spots
Q_formation = 30.0  # Reduced from 50.0

for k in range(N):
    ees_k = []
    bases_k = []
    for i in range(NUM_ROBOTS):
        idx_base = i * n_states_single
        idx_arm = idx_base + 2
        ee_k = fk_ee_sym(X[idx_base:idx_base+2, k], X[idx_arm:idx_arm+n_arm_joints, k], arm_link_params)
        ees_k.append(ee_k)
        bases_k.append(X[idx_base:idx_base+2, k])
    
    # Object centroid for 5 robots
    obj_cent_k = sum(ees_k) / float(NUM_ROBOTS)
    error_k = obj_cent_k - X_ref_p[:, k]
    cost += ca.mtimes([error_k.T, Q_obj, error_k])
    cost += ca.mtimes([U[:, k].T, R, U[:, k]])
    
    # Formation cost: penalize bases deviating from expected positions relative to centroid
    for i in range(NUM_ROBOTS):
        expected_base = obj_cent_k + initial_base_offsets[i]
        base_error = bases_k[i] - expected_base
        cost += Q_formation * ca.dot(base_error, base_error)
    
    # Progress incentive: reward forward motion (positive x velocity)
    # Compute average base velocity in x direction
    avg_vx = sum([U[i*n_controls_single, k] for i in range(NUM_ROBOTS)]) / NUM_ROBOTS
    progress_reward = 5.0  # Reward for moving forward
    cost -= progress_reward * avg_vx  # Negative because we minimize cost

ees_N = []
for i in range(NUM_ROBOTS):
    idx_base = i * n_states_single
    idx_arm = idx_base + 2
    ee_N = fk_ee_sym(X[idx_base:idx_base+2, N], X[idx_arm:idx_arm+n_arm_joints, N], arm_link_params)
    ees_N.append(ee_N)

obj_cent_N = sum(ees_N) / float(NUM_ROBOTS)
error_N = obj_cent_N - X_ref_p[:, N]
cost += ca.mtimes([error_N.T, Q_terminal_obj, error_N])

cost += cost_cbf_slack * ca.sum1(ca.sum2(cbf_slack**2))
cost += W_rigidity_slack * ca.sum1(ca.sum2(slack_rigidity**2))

opti.minimize(cost)

# ==================== VELOCITY-BASED CBF CONSTRAINTS ====================
print("\n" + "="*80)
print("VELOCITY-BASED CBF FOR 5-ROBOT PENTAGON")
print("="*80)
print("Complete safety system with:")
print("  1. Inter-robot collision avoidance (10 pairs)")
print("  2. Robot bases to STATIC obstacles")
print("  3. Object (EEs) to STATIC obstacles")
print("  4. Robot bases to DYNAMIC obstacles")
print("  5. Object (EEs) to DYNAMIC obstacles")
print("  6. Angular non-collinearity")
print("  + RIGID FORMATION via all pairwise EE + BASE distances")
print("="*80 + "\n")

cbf_idx = 0

for k in range(N):
    # Get all states and velocities
    ees_k = []
    bases_k = []
    base_vels_k = []
    ee_vels_k = []
    
    for i in range(NUM_ROBOTS):
        idx_base = i * n_states_single
        idx_arm = idx_base + 2
        base_k = X[idx_base:idx_base+2, k]
        arm_k = X[idx_arm:idx_arm+n_arm_joints, k]
        ee_k = fk_ee_sym(base_k, arm_k, arm_link_params)
        
        # Base velocity
        v_base_k = U[i*n_controls_single:i*n_controls_single+2, k]
        
        # EE velocity (using Jacobian)
        q1, d2, q3 = arm_k[0], arm_k[1], arm_k[2]
        L_total = arm_link_params[0] + d2
        
        # Jacobian columns
        J_q1 = ca.vertcat(-L_total * ca.sin(q1) - arm_link_params[1] * ca.sin(q1 + q3),
                          L_total * ca.cos(q1) + arm_link_params[1] * ca.cos(q1 + q3))
        J_d2 = ca.vertcat(ca.cos(q1), ca.sin(q1))
        J_q3 = ca.vertcat(-arm_link_params[1] * ca.sin(q1 + q3),
                          arm_link_params[1] * ca.cos(q1 + q3))
        
        # EE velocity: v_ee = v_base + J_arm * u_arm
        u_arm_k = U[i*n_controls_single+2:i*n_controls_single+5, k]
        v_ee_k = v_base_k + J_q1 * u_arm_k[0] + J_d2 * u_arm_k[1] + J_q3 * u_arm_k[2]
        
        ees_k.append(ee_k)
        bases_k.append(base_k)
        base_vels_k.append(v_base_k)
        ee_vels_k.append(v_ee_k)
    
    # Compute object centroid and its velocity for 5 robots
    obj_cent_k = sum(ees_k) / float(NUM_ROBOTS)
    obj_cent_vel_k = sum(ee_vels_k) / float(NUM_ROBOTS)
    
    # ===== 1. VELOCITY-BASED INTER-ROBOT CBF =====
    cbf_idx = 0
    for i in range(NUM_ROBOTS):
        for j in range(i + 1, NUM_ROBOTS):
            p_diff = bases_k[i] - bases_k[j]
            h_inter = ca.dot(p_diff, p_diff) - d_min_inter**2
            v_diff = base_vels_k[i] - base_vels_k[j]
            h_dot = 2 * ca.dot(p_diff, v_diff)
            cbf_condition = h_dot + gamma_cbf_velocity * h_inter
            
            dist_ij = ca.norm_2(p_diff)
            gate_active = ca.if_else(dist_ij < GATE_INTER_ROBOT, 1.0, 0.0)
            
            opti.subject_to(gate_active * cbf_condition + cbf_slack[cbf_idx, k] >= 0)
            cbf_idx += 1
    
    # ===== 2. VELOCITY-BASED ROBOT BASE TO STATIC OBSTACLES =====
    for i in range(NUM_ROBOTS):
        for obs_idx, obs_center in enumerate(static_obs_center_p):
            p_diff = bases_k[i] - obs_center
            safe_dist = static_obstacle_radius + d_min_obstacle_base
            h_obs = ca.dot(p_diff, p_diff) - safe_dist**2
            h_dot = 2 * ca.dot(p_diff, base_vels_k[i])
            cbf_condition = h_dot + gamma_cbf_velocity * h_obs
            
            opti.subject_to(cbf_condition + cbf_slack[cbf_idx, k] >= 0)
            cbf_idx += 1
    
    # ===== 3. VELOCITY-BASED EE TO STATIC OBSTACLES =====
    for i in range(NUM_ROBOTS):
        for obs_idx, obs_center in enumerate(static_obs_center_p):
            p_diff = ees_k[i] - obs_center
            safe_dist = static_obstacle_radius + d_min_obstacle_object
            h_ee_obs = ca.dot(p_diff, p_diff) - safe_dist**2
            h_dot_ee = 2 * ca.dot(p_diff, ee_vels_k[i])
            cbf_condition = h_dot_ee + gamma_cbf_velocity * h_ee_obs
            
            opti.subject_to(cbf_condition + cbf_slack[cbf_idx, k] >= 0)
            cbf_idx += 1
    
    # ===== 4. VELOCITY-BASED ROBOT BASE TO DYNAMIC OBSTACLES =====
    for i in range(NUM_ROBOTS):
        for obs_idx in range(num_dynamic_obstacles):
            obs_center = dynamic_obs_center_p[obs_idx]
            obs_vel = dynamic_obs_vel_p[obs_idx]
            
            p_diff = bases_k[i] - obs_center
            safe_dist = dynamic_obstacle_radius + d_min_obstacle_base
            h_obs = ca.dot(p_diff, p_diff) - safe_dist**2
            v_relative = base_vels_k[i] - obs_vel
            h_dot = 2 * ca.dot(p_diff, v_relative)
            cbf_condition = h_dot + gamma_cbf_velocity * h_obs
            
            opti.subject_to(cbf_condition + cbf_slack[cbf_idx, k] >= 0)
            cbf_idx += 1
    
    # ===== 5. VELOCITY-BASED EE TO DYNAMIC OBSTACLES =====
    for i in range(NUM_ROBOTS):
        for obs_idx in range(num_dynamic_obstacles):
            obs_center = dynamic_obs_center_p[obs_idx]
            obs_vel = dynamic_obs_vel_p[obs_idx]
            
            p_diff = ees_k[i] - obs_center
            safe_dist = dynamic_obstacle_radius + d_min_obstacle_object
            h_ee_obs = ca.dot(p_diff, p_diff) - safe_dist**2
            v_relative = ee_vels_k[i] - obs_vel
            h_dot_ee = 2 * ca.dot(p_diff, v_relative)
            cbf_condition = h_dot_ee + gamma_cbf_velocity * h_ee_obs
            
            opti.subject_to(cbf_condition + cbf_slack[cbf_idx, k] >= 0)
            cbf_idx += 1
    
    # ===== 6. ANGULAR NON-COLLINEARITY =====
    for i in range(NUM_ROBOTS):
        i_left = (i - 1) % NUM_ROBOTS
        i_right = (i + 1) % NUM_ROBOTS
        
        vertex_left = ees_k[i_left]
        vertex_right = ees_k[i_right]
        
        h_angular = compute_proper_angular_constraint(
            bases_k[i], ees_k[i], vertex_left, vertex_right,
            robot_radius, theta_min_collinearity
        )
        
        gamma_angular = 15.0
        cbf_condition = h_angular + gamma_angular * dt * h_angular
        
        # opti.subject_to(cbf_condition + cbf_slack[cbf_idx, k] >= 0)
        cbf_idx += 1

# ==================== RIGIDITY CONSTRAINTS (ALL PAIRWISE for true rigidity) ====================
# For a rigid pentagon, we need ALL pairwise distances constrained
# 5 robots -> 10 pairs (5 edges + 5 diagonals)

# EE rigidity pairs
rigidity_pairs = []
initial_pair_lengths = []
for i in range(NUM_ROBOTS):
    for j in range(i + 1, NUM_ROBOTS):
        rigidity_pairs.append((i, j))
        d_ij_0 = np.linalg.norm(initial_ees[i] - initial_ees[j])
        initial_pair_lengths.append(d_ij_0)

n_rigidity_pairs = len(rigidity_pairs)  # Should be 10 for pentagon
print(f"EE Rigidity pairs (edges + diagonals): {n_rigidity_pairs}")
for idx, (i, j) in enumerate(rigidity_pairs):
    print(f"  Pair {idx}: robot {i} <-> robot {j}, initial EE dist = {initial_pair_lengths[idx]:.4f}m")

# BASE rigidity pairs - CRITICAL for preventing outward swings
initial_bases = np.array([x0[i*n_states_single:i*n_states_single+2] for i in range(NUM_ROBOTS)])
base_rigidity_pairs = []
initial_base_pair_lengths = []
for i in range(NUM_ROBOTS):
    for j in range(i + 1, NUM_ROBOTS):
        base_rigidity_pairs.append((i, j))
        d_ij_0 = np.linalg.norm(initial_bases[i] - initial_bases[j])
        initial_base_pair_lengths.append(d_ij_0)

print(f"\nBASE Rigidity pairs: {len(base_rigidity_pairs)}")
for idx, (i, j) in enumerate(base_rigidity_pairs):
    print(f"  Pair {idx}: robot {i} <-> robot {j}, initial BASE dist = {initial_base_pair_lengths[idx]:.4f}m")

rigidity_tol = 0.008  # Slightly relaxed for maneuverability
base_rigidity_tol = 0.03  # Relaxed from 0.02 to allow more formation flexibility

# Add rigidity constraints for ALL EE pairs
for k in range(N):
    ees_k_list = []
    bases_k_list = []
    for i in range(NUM_ROBOTS):
        idx_base = i * n_states_single
        idx_arm = idx_base + 2
        ee = fk_ee_sym(X[idx_base:idx_base+2, k], X[idx_arm:idx_arm+n_arm_joints, k], arm_link_params)
        ees_k_list.append(ee)
        bases_k_list.append(X[idx_base:idx_base+2, k])
    
    # EE Rigidity for ALL pairs (edges AND diagonals)
    for pair_idx, (i, j) in enumerate(rigidity_pairs):
        d_ij_k = ca.norm_2(ees_k_list[i] - ees_k_list[j])
        d_ij_0 = initial_pair_lengths[pair_idx]
        
        opti.subject_to(d_ij_k - slack_rigidity[pair_idx, k] <= d_ij_0 * (1 + rigidity_tol))
        opti.subject_to(d_ij_k + slack_rigidity[pair_idx, k] >= d_ij_0 * (1 - rigidity_tol))
    
    # BASE Rigidity for ALL pairs - prevents robots from swinging outward
    for pair_idx, (i, j) in enumerate(base_rigidity_pairs):
        d_base_ij_k = ca.norm_2(bases_k_list[i] - bases_k_list[j])
        d_base_ij_0 = initial_base_pair_lengths[pair_idx]
        
        # Hard constraints - no slack for base formation
        opti.subject_to(d_base_ij_k <= d_base_ij_0 * (1 + base_rigidity_tol))
        opti.subject_to(d_base_ij_k >= d_base_ij_0 * (1 - base_rigidity_tol))

# ==================== SOLVER CONFIGURATION ====================
opts = {
    'ipopt.print_level': 0,
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.max_iter': 3000,  # Increased for longer horizon
    'ipopt.tol': 1e-6,
    'ipopt.acceptable_tol': 1e-5,
    'ipopt.acceptable_iter': 20,  # Increased
    'print_time': 0
}
opti.solver('ipopt', opts)

# ==================== SIMULATION ====================
x_curr = x0.copy()
last_X = np.tile(x_curr.reshape(-1, 1), (1, N + 1))
last_U = np.zeros((n_controls, N))
last_cbf_slack = np.zeros((n_cbf_constraints, N))
last_rigidity_slack = np.zeros((n_rigidity, N))

hist_x, hist_u = [x_curr.copy()], []
hist_solver_status, hist_max_slack = [], []
hist_dynamic_obs_positions = []
solve_times, failure_count = [], 0

print("Starting MPC simulation with 5 ROBOTS in PENTAGON formation...")
print(f"Total timesteps: {time_steps}")
print(f"Horizon: {N} steps")
print(f"Base-obstacle clearance: {d_min_obstacle_base}m")
print(f"Object-obstacle clearance: {d_min_obstacle_object}m")
print(f"Base-object clearance: {d_min_base_object}m")
print("-" * 80)

for step_idx in range(time_steps):
    t_start = time.time()
    
    # Reference window
    X_ref_win = X_ref_obj_full[:, step_idx:step_idx + N + 1]
    if X_ref_win.shape[1] < N + 1:
        last_col = X_ref_win[:, -1:]
        X_ref_win = np.hstack([X_ref_win, np.tile(last_col, (1, N + 1 - X_ref_win.shape[1]))])
    
    # Set parameters
    opti.set_value(x0_p, x_curr)
    opti.set_value(X_ref_p, X_ref_win)
    
    # Set STATIC obstacle parameters
    for oi, obs_c in enumerate(static_obstacle_centers):
        opti.set_value(static_obs_center_p[oi], obs_c)
    
    # Set DYNAMIC obstacle parameters
    current_dyn_obs_pos = []
    for oi in range(num_dynamic_obstacles):
        pos = all_obstacle_trajs[oi][:, step_idx]
        vel = get_velocity_from_trajectory(all_obstacle_trajs[oi], step_idx, dt)
        
        opti.set_value(dynamic_obs_center_p[oi], pos)
        opti.set_value(dynamic_obs_vel_p[oi], vel)
        current_dyn_obs_pos.append(pos)
    
    hist_dynamic_obs_positions.append(current_dyn_obs_pos)
    
    # Warm start
    opti.set_initial(X, last_X)
    opti.set_initial(U, last_U)
    opti.set_initial(cbf_slack, last_cbf_slack)
    opti.set_initial(slack_rigidity, last_rigidity_slack)
    
    # Solve
    try:
        sol = opti.solve()
        X_sol = sol.value(X)
        U_sol = sol.value(U)
        cbf_slack_sol = sol.value(cbf_slack)
        rigidity_slack_sol = sol.value(slack_rigidity)
        
        status = "success"
        u_apply = U_sol[:, 0]
        
        last_X = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
        last_U = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])
        last_cbf_slack = np.hstack([cbf_slack_sol[:, 1:], cbf_slack_sol[:, -1:]])
        last_rigidity_slack = np.hstack([rigidity_slack_sol[:, 1:], rigidity_slack_sol[:, -1:]])
        
    except Exception as e:
        try:
            X_sol = opti.debug.value(X)
            U_sol = opti.debug.value(U)
            cbf_slack_sol = opti.debug.value(cbf_slack)
            rigidity_slack_sol = opti.debug.value(slack_rigidity)
            
            status = "recovered"
            u_apply = U_sol[:, 0]
            
            last_X = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
            last_U = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])
            last_cbf_slack = np.hstack([cbf_slack_sol[:, 1:], cbf_slack_sol[:, -1:]])
            last_rigidity_slack = np.hstack([rigidity_slack_sol[:, 1:], rigidity_slack_sol[:, -1:]])
            
        except:
            print(f"\n[Step {step_idx}] Solver failed: {e}")
            status = "failed"
            failure_count += 1
            u_apply = np.zeros(n_controls)
            cbf_slack_sol = np.zeros((n_cbf_constraints, N))
    
    # Apply control
    for i in range(NUM_ROBOTS):
        idx = i * n_states_single
        idx_u = i * n_controls_single
        x_curr[idx:idx+2] += dt * u_apply[idx_u:idx_u+2]
        x_curr[idx+2:idx+n_states_single] += dt * u_apply[idx_u+2:idx_u+n_controls_single]
    
    x_curr = np.clip(x_curr, lbq, ubq)
    
    # Logging
    t_solve = time.time() - t_start
    solve_times.append(t_solve)
    hist_x.append(x_curr.copy())
    hist_u.append(u_apply.copy())
    hist_solver_status.append(status)
    hist_max_slack.append(np.max(cbf_slack_sol))
    
    # Print progress
    if step_idx % 10 == 0 or step_idx == time_steps - 1:
        print(f"Step {step_idx:3d}/{time_steps} | "
              f"Solve: {t_solve:.3f}s | "
              f"Status: {status:9s} | "
              f"Slack: {hist_max_slack[-1]:.4f}")

print("\n" + "="*80)
print("SIMULATION COMPLETE")
print("="*80)
print(f"Total solve time: {sum(solve_times):.2f}s")
print(f"Average solve time: {np.mean(solve_times):.3f}s")
print(f"Failures: {failure_count}/{time_steps}")
print(f"Max CBF slack: {max(hist_max_slack):.5f}")
print("="*80 + "\n")

# ==================== ANIMATION ====================
print("Creating animation with 5-robot pentagon formation...")

fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))

ax1.set_xlim(-1, 16)
ax1.set_ylim(-1, 6)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('5-Robot Pentagon Formation with Static & Dynamic Obstacles')

# Draw STATIC obstacles
for center in static_obstacle_centers:
    hex_verts = create_hexagon_numpy(center, static_obstacle_radius)
    polygon = Polygon(hex_verts.T, fill=True, facecolor='red', edgecolor='darkred', 
                     alpha=0.3, linewidth=2)
    ax1.add_patch(polygon)
    ax1.text(center[0], center[1], 'STATIC', ha='center', va='center', 
            fontsize=8, color='darkred', weight='bold')

# DYNAMIC obstacles (will be animated)
dynamic_obs_patches = []
dynamic_obs_trails = []
for i in range(num_dynamic_obstacles):
    hex_verts = create_hexagon_numpy([0, 0], dynamic_obstacle_radius)
    polygon = Polygon(hex_verts.T, fill=True, facecolor='orange', edgecolor='darkorange', 
                     alpha=0.5, linewidth=2)
    ax1.add_patch(polygon)
    dynamic_obs_patches.append(polygon)
    
    trail_line, = ax1.plot([], [], 'orange', alpha=0.3, linewidth=1, linestyle='--')
    dynamic_obs_trails.append(trail_line)

# Robots - 5 colors for 5 robots
robot_circles = []
robot_arms = []
robot_triangles = []
colors = ['blue', 'green', 'purple', 'cyan', 'magenta']

for i in range(NUM_ROBOTS):
    circle = Circle((0, 0), robot_radius, fill=False, edgecolor=colors[i], linewidth=2)
    ax1.add_patch(circle)
    robot_circles.append(circle)
    
    arm_line, = ax1.plot([], [], 'o-', color=colors[i], linewidth=2, markersize=4)
    robot_arms.append(arm_line)
    
    tri = Polygon([[0,0]], fill=False, edgecolor=colors[i], linewidth=1.5, linestyle='--', alpha=0.6)
    ax1.add_patch(tri)
    robot_triangles.append(tri)

obj_polygon = Polygon([[0,0]], fill=False, edgecolor='black', linewidth=3, label='Pentagon Object')
ax1.add_patch(obj_polygon)

ax1.plot(X_ref_obj_full[0, :], X_ref_obj_full[1, :], 'k--', alpha=0.5, linewidth=1.5, label='Reference')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', edgecolor='darkred', alpha=0.3, label='Static Obstacle'),
    Patch(facecolor='orange', edgecolor='darkorange', alpha=0.5, label='Dynamic Obstacle'),
    Patch(facecolor='none', edgecolor='black', linewidth=3, label='Pentagon Object')
]
ax1.legend(handles=legend_elements, loc='upper left')

time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def init():
    for i in range(NUM_ROBOTS):
        robot_circles[i].center = (0, 0)
        robot_arms[i].set_data([], [])
        robot_triangles[i].set_xy([[0,0]])
    obj_polygon.set_xy([[0,0]])
    time_text.set_text('')
    for patch in dynamic_obs_patches:
        patch.set_xy([[0,0]])
    for trail in dynamic_obs_trails:
        trail.set_data([], [])
    return robot_circles + robot_arms + robot_triangles + [obj_polygon, time_text] + dynamic_obs_patches + dynamic_obs_trails

def animate(frame):
    x_curr_frame = hist_x[frame]
    
    # Update robots
    curr_ees_frame = []
    for i in range(NUM_ROBOTS):
        idx = i * n_states_single
        base = x_curr_frame[idx:idx+2]
        arm = x_curr_frame[idx+2:idx+n_states_single]
        ee = fk_ee_world_np(base, arm, arm_link_params)
        curr_ees_frame.append(ee)
        
        robot_circles[i].center = (base[0], base[1])
        
        q1, d2, q3 = arm
        j1_pos = base + np.array([(L_arm1 + d2) * np.cos(q1), 
                                  (L_arm1 + d2) * np.sin(q1)])
        robot_arms[i].set_data([base[0], j1_pos[0], ee[0]], 
                               [base[1], j1_pos[1], ee[1]])
        
        vec_RA = ee - base
        d_RA = np.linalg.norm(vec_RA) + 1e-6
        theta = np.arcsin(min(robot_radius / d_RA, 0.999))
        u_RA = vec_RA / d_RA
        vec_RF = (d_RA - robot_radius * np.cos(theta)) * u_RA
        F = base + vec_RF
        d_perp = robot_radius * np.sin(theta)
        u_perp = np.array([-u_RA[1], u_RA[0]])
        B = F + d_perp * u_perp
        C = F - d_perp * u_perp
        robot_triangles[i].set_xy(np.array([ee, B, C]))
    
    # Pentagon object from 5 end-effectors
    obj_polygon.set_xy(np.array(curr_ees_frame))
    
    # Update dynamic obstacles
    if frame < len(hist_dynamic_obs_positions):
        for i, patch in enumerate(dynamic_obs_patches):
            pos = hist_dynamic_obs_positions[frame][i]
            hex_verts = create_hexagon_numpy(pos, dynamic_obstacle_radius)
            patch.set_xy(hex_verts.T)
            
            trail_x = [hist_dynamic_obs_positions[f][i][0] for f in range(max(0, frame-30), frame+1)]
            trail_y = [hist_dynamic_obs_positions[f][i][1] for f in range(max(0, frame-30), frame+1)]
            dynamic_obs_trails[i].set_data(trail_x, trail_y)
    
    time_text.set_text(f'Time: {frame * dt:.1f}s\nStep: {frame}/{len(hist_x)-1}\n'
                      f'5 Robots - Pentagon Formation')
    
    return robot_circles + robot_arms + robot_triangles + [obj_polygon, time_text] + dynamic_obs_patches + dynamic_obs_trails

anim = FuncAnimation(fig, animate, init_func=init, frames=len(hist_x), 
                    interval=50, blit=True, repeat=True)

output_file = '/home/manan/Desktop/IITB/honors/honors/5_robot_pentagon_formation_sinusoidal_trajectory.mp4'
try:
    writer = FFMpegWriter(fps=20, bitrate=2000)
    anim.save(output_file, writer=writer)
    print(f"Animation saved to: {output_file}")
except Exception as e:
    print(f"Could not save animation: {e}")

plt.tight_layout()
plt.show()

print("\n5-Robot Pentagon Formation Complete!")
print("Robots grasp at midpoints of pentagon edges, maintaining rigid formation.")