# Game
Create a non-player character (NPC) that learns to navigate a simple grid-based environment using reinforcement learning (RL). Here we used Q-Learning, , a popular RL algorithm, to teach the NPC to reach a target while avoiding obstacles.

Explanation of the Code:

- Environment Setup:
A grid environment is defined with a target position and obstacles. The agent (NPC) starts at the top-left corner.

- Q-table:
A 3D numpy array stores the Q-values for each state-action pair.

- Q-Learning Algorithm:
The agent learns by interacting with the environment, updating Q-values based on rewards and the maximum future Q-value.

- Reward System:
Positive reward for reaching the target, negative reward for hitting obstacles, and a small penalty for each move to encourage efficiency.

- Testing:
After training, the agent follows the policy it has learned (using the Q-table) to navigate the environment.
