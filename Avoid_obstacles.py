import numpy as np
import random

# Define the grid environment
GRID_SIZE = 5
TARGET_POSITION = (4, 4)
OBSTACLES = [(2, 2), (3, 1)]

# Define actions (up, down, left, right)
ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

# Initialize the Q-table
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Hyperparameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate
EPISODES = 1000

# Helper function: Check if position is valid
def is_valid_position(position):
    x, y = position
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and position not in OBSTACLES

# Function to choose an action based on the epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(list(ACTIONS.keys()))  # Explore
    else:
        x, y = state
        return np.argmax(Q_table[x, y])  # Exploit

# Training the NPC using Q-learning
for episode in range(EPISODES):
    state = (0, 0)  # Start position
    total_reward = 0

    while state != TARGET_POSITION:
        x, y = state

        # Choose an action
        action = choose_action(state)
        
        # Calculate new position
        new_state = (x + ACTIONS[action][0], y + ACTIONS[action][1])

        # Check if the new position is valid
        if not is_valid_position(new_state):
            new_state = state  # Stay in the same position

        # Reward system
        if new_state == TARGET_POSITION:
            reward = 100  # Reaching the target
        elif new_state in OBSTACLES:
            reward = -100  # Hitting an obstacle
        else:
            reward = -1  # Small penalty for each move

        # Update Q-value using the Q-learning formula
        old_value = Q_table[x, y, action]
        next_max = np.max(Q_table[new_state[0], new_state[1]])
        Q_table[x, y, action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

        state = new_state
        total_reward += reward

    # Optional: Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Test the learned policy
def test_agent():
    state = (0, 0)
    path = [state]

    while state != TARGET_POSITION:
        x, y = state
        action = np.argmax(Q_table[x, y])
        state = (x + ACTIONS[action][0], y + ACTIONS[action][1])

        if not is_valid_position(state):
            break

        path.append(state)

    return path

# Run the test
test_path = test_agent()
print("Learned Path:", test_path)
