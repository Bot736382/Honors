import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO

class RobotArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 6 Degrees of Freedom actions
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        # Observation: 6 joint states + 3 target coords
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handle Gymnasium seeding
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load a 6-DOF Kuka Arm
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        self.target_pos = np.array([0.4, 0.3, 0.5])
        
        # Create a visual marker for the target
        p.loadURDF("sphere_small.urdf", self.target_pos, globalScaling=2)
        
        self.joint_angles = np.zeros(6)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.joint_angles, self.target_pos]).astype(np.float32)

    def step(self, action):
        # Apply action to joint angles
        self.joint_angles = np.clip(self.joint_angles + action * 0.05, -2.9, 2.9)
        
        for i in range(6):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, self.joint_angles[i])
        
        p.stepSimulation()
        
        # Calculate End Effector distance to target
        ee_state = p.getLinkState(self.robot_id, 6)
        ee_pos = np.array(ee_state[0])
        dist = np.linalg.norm(ee_pos - self.target_pos)
        
        reward = -dist # Dense reward: get closer = higher reward
        terminated = bool(dist < 0.05) # Goal reached
        truncated = False # Time limit (could be set to 500 steps)
        
        return self._get_obs(), reward, terminated, truncated, {}

# --- Training Block ---
env = RobotArmEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_arm_logs/")

print("Training... (Close the PyBullet window to stop)")
model.learn(total_timesteps=20000)

# --- Quick Test of the Trained Model ---
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()