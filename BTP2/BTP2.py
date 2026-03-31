import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt


def save_epoch_bot_plot(epoch_idx, bot_trajectory, goal_x, output_dir):
    """Save one plot per epoch showing paths for all 5 bots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange", "tab:purple"]

    for i in range(bot_trajectory.shape[1]):
        x_vals = bot_trajectory[:, i, 0].tolist()
        y_vals = bot_trajectory[:, i, 1].tolist()
        ax.plot(x_vals, y_vals, color=colors[i], linewidth=2, label=f"Bot {i + 1}")
        ax.scatter(x_vals[0], y_vals[0], color=colors[i], marker="o", s=35)
        ax.scatter(x_vals[-1], y_vals[-1], color=colors[i], marker="x", s=35)

    ax.axvline(goal_x, color="black", linestyle="--", linewidth=1.5, label="Goal Line")
    ax.set_title(f"Bot trajectories at epoch {epoch_idx}")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"epoch_{epoch_idx:03d}.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

# =========================
# 1. Simple Environment
# =========================
class SimpleEnv:
    def __init__(self):
        self.num_bots = 5
        self.goal_x = 10.0
        self.bot_positions = None
        self.bot_history = []
        self.state = torch.tensor([2.0, 1.0])  # [distance_to_goal, obstacle]

    def reset(self):
        y_offsets = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.bot_positions = torch.stack([torch.zeros(self.num_bots), y_offsets], dim=1)
        self.bot_history = [self.bot_positions.clone()]
        avg_distance_to_goal = self.goal_x - torch.mean(self.bot_positions[:, 0])
        self.state = torch.tensor([avg_distance_to_goal, 1.0])
        return self.state.clone()

    def step(self, action):
        # action: 0,1,2 controls how much all bots move in +x direction
        move_x = torch.tensor([0.5, 0.2, 0.35])[action]
        lateral_noise = torch.tensor([random.uniform(-0.03, 0.03) for _ in range(self.num_bots)])

        self.bot_positions[:, 0] += move_x
        self.bot_positions[:, 1] += lateral_noise
        self.bot_history.append(self.bot_positions.clone())

        avg_distance_to_goal = self.goal_x - torch.mean(self.bot_positions[:, 0])
        # Build a fresh state tensor to avoid in-place modifications that break autograd.
        obstacle_risk = max(0.0, self.state[1].item() + random.uniform(-0.1, 0.1))
        self.state = torch.tensor([avg_distance_to_goal.item(), obstacle_risk])

        # lower distance and obstacle risk means lower cost
        cost = self.state[0] + 0.5 * self.state[1]

        done = self.state[0] <= 0.1
        return self.state.clone(), cost, done

    def get_bot_trajectory(self):
        return torch.stack(self.bot_history, dim=0)


# =========================
# 2. Policy Network (Actor)
# =========================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return F.softmax(logits, dim=-1)


# =========================
# 3. Value Network (Critic)
# =========================
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# =========================
# 4. Setup
# =========================
env = SimpleEnv()

policy = PolicyNet(state_dim=2, action_dim=3)
value = ValueNet(state_dim=2)

policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value.parameters(), lr=1e-3)

gamma = 0.9


# =========================
# 5. Training Loop
# =========================
for epoch in range(1):

    state = env.reset()
    total_cost = 0

    for step in range(50):

        # -------------------------
        # Policy forward pass
        # -------------------------
        probs = policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # -------------------------
        # Environment step
        # -------------------------
        next_state, cost, done = env.step(action.item())

        # -------------------------
        # Value estimates
        # -------------------------
        V_s = value(state)
        V_next = value(next_state).detach()

        # -------------------------
        # Bellman target
        # -------------------------
        target = cost + gamma * V_next

        # -------------------------
        # Value loss
        # -------------------------
        value_loss = (V_s - target) ** 2

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # -------------------------
        # Advantage
        # -------------------------
        advantage = (target - V_s).detach()

        # -------------------------
        # Policy loss
        # -------------------------
        log_prob = dist.log_prob(action)
        policy_loss = -log_prob * advantage

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        state = next_state
        total_cost += cost

        if done:
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Total Cost: {total_cost:.2f}")

    epoch_trajectory = env.get_bot_trajectory()
    plots_dir = os.path.join(os.path.dirname(__file__), "epoch_bot_plots")
    save_epoch_bot_plot(epoch, epoch_trajectory, env.goal_x, plots_dir)


# =========================
# 6. Save Trained Models
# =========================
save_path = os.path.join(os.path.dirname(__file__), "btp2_checkpoint.pth")
torch.save(
    {
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "value_optimizer_state_dict": value_optimizer.state_dict(),
        "gamma": gamma,
    },
    save_path,
)
print(f"Saved checkpoint to: {save_path}")