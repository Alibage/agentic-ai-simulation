
import numpy as np
import matplotlib.pyplot as plt

# Simulation settings
T = 100  # time steps
num_agents = 3  # number of agents

# Initialize arrays
goal_progress = np.zeros((num_agents, T))
knowledge_level = np.zeros((num_agents, T))
autonomy_level = np.zeros((num_agents, T))
human_trust = np.zeros((num_agents, T))
complexity = np.linspace(0.2, 0.8, T)  # increasing task complexity
intervention_rate = 0.3  # constant intervention rate
goal_alignment = np.ones((num_agents, T))  # 1 = aligned, 0 = misaligned

# Traits for agents
risk_tolerance = [0.6, 0.3, 0.8]
communication_ability = [0.7, 0.5, 0.9]

# Initial conditions
goal_progress[:, 0] = 0
knowledge_level[:, 0] = [0.3, 0.5, 0.7]
autonomy_level[:, 0] = [0.5, 0.3, 0.6]
human_trust[:, 0] = [0.6, 0.7, 0.4]

# Parameters
learning_rate = 0.01
trust_decay = 0.01
trust_gain = 0.02
autonomy_gain = 0.01
autonomy_loss = 0.02

# Task switching and misalignment events
switch_task_step = 40
misalignment_step = 60

# Simulation loop
for agent in range(num_agents):
    for t in range(1, T):
        if t == switch_task_step:
            goal_progress[agent, t-1] = max(goal_progress[agent, t-1] - 20, 0)

        if t >= misalignment_step:
            goal_alignment[agent, t] = 0.5

        effectiveness = (autonomy_level[agent, t-1] *
                         knowledge_level[agent, t-1] *
                         (1 - complexity[t]) *
                         goal_alignment[agent, t])

        progress = effectiveness * 10
        goal_progress[agent, t] = min(100, goal_progress[agent, t-1] + progress)
        knowledge_level[agent, t] = min(1.0, knowledge_level[agent, t-1] + learning_rate * effectiveness)

        if effectiveness > 0.2:
            human_trust[agent, t] = min(1.0, human_trust[agent, t-1] +
                                        trust_gain * effectiveness * communication_ability[agent])
        else:
            human_trust[agent, t] = max(0.0, human_trust[agent, t-1] - trust_decay)

        if human_trust[agent, t] > 0.5:
            autonomy_level[agent, t] = min(1.0, autonomy_level[agent, t-1] + autonomy_gain)
        else:
            autonomy_level[agent, t] = max(0.1, autonomy_level[agent, t-1] - autonomy_loss * intervention_rate)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(16, 18), dpi=300)
agent_colors = ['tab:blue', 'tab:green', 'tab:red']
agent_traits = [
    f"Agent 1 (RiskTol: {risk_tolerance[0]}, Comm: {communication_ability[0]})",
    f"Agent 2 (RiskTol: {risk_tolerance[1]}, Comm: {communication_ability[1]})",
    f"Agent 3 (RiskTol: {risk_tolerance[2]}, Comm: {communication_ability[2]})"
]

titles = ["Goal Progress Over Time",
          "Knowledge Level Over Time",
          "Autonomy Level Over Time",
          "Human Trust Over Time"]

y_labels = ["Goal Progress (%)",
            "Knowledge Level (0–1)",
            "Autonomy Level (0–1)",
            "Human Trust (0–1)"]

data_series = [goal_progress, knowledge_level, autonomy_level, human_trust]

for i, ax in enumerate(axs):
    for agent in range(num_agents):
        ax.plot(data_series[i][agent],
                label=agent_traits[agent], color=agent_colors[agent])

        if i == 0:
            ax.axvline(40, color='gray', linestyle='--', linewidth=1)
            ax.text(42, goal_progress[agent, 40] + 2,
                    f"Task Switch A{agent+1}", color=agent_colors[agent], fontsize=9)
            ax.axvline(60, color='black', linestyle='--', linewidth=1)
            ax.text(62, goal_progress[agent, 60] + 2,
                    f"Misalign A{agent+1}", color=agent_colors[agent], fontsize=9)

    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_ylabel(y_labels[i])
    ax.set_xlabel("Time Step")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 100 if i == 0 else 1)
    ax.grid(True)
    ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig("Agentic_AI_Simulation_Full_Code.png")
plt.show()
