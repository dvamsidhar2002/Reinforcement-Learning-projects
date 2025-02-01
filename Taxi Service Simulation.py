import numpy as np
import gym
import random
import time  # Added for visualization delay

# Try importing pygame for visualization
try:
    import pygame
except ImportError:
    raise ImportError("Pygame is not installed. Please run 'pip install pygame gym[toy_text]'")

# Q-Table Initialization Function
def initialize_q_table(state_space_size, action_space_size):
    return np.zeros((state_space_size, action_space_size))

# Epsilon Decay Function
def decay_epsilon(epsilon, decay_rate, episode, min_epsilon):
    return max(min_epsilon, epsilon * (1 - decay_rate))  # Adjusted decay for smoother reduction

# Training Function
def train_agent(env, Q_table, alpha, gamma, epsilon, decay_rate, min_epsilon, episodes, max_steps):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        for step in range(max_steps):
            # Exploration-Exploitation Trade-off
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q_table[state, :])  # Exploit

            # Take Action
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q-Table
            Q_table[state, action] = Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[new_state, :]) - Q_table[state, action])

            state = new_state

            if done:
                break

        # Epsilon Decay
        epsilon = decay_epsilon(epsilon, decay_rate, episode, min_epsilon)

        # Display progress every 1 episodes
        if (episode + 1) % 1 == 0:
            print(f"Completed {episode + 1}/{episodes} episodes")

    return Q_table, epsilon

# Testing Function
def test_agent(env, Q_table, tests):
    total_epochs, total_penalties = 0, 0
    for _ in range(tests):
        state, _ = env.reset()
        epochs, penalties = 0, 0
        done = False

        while not done:
            action = np.argmax(Q_table[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    return total_epochs / tests, total_penalties / tests

# Visualization Function
def visualize_agent(env, Q_table):
    state, _ = env.reset()
    done = False
    print("\nVisualizing Taxi's Path:")

    try:
        time.sleep(0.5)  # Reduced initial delay for quicker visualization start

        while not done:
            env.render()
            action = np.argmax(Q_table[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            time.sleep(0.3)  # Reduced delay for faster visualization

    except gym.error.DependencyNotInstalled:
        print("Rendering requires pygame. Please install it using 'pip install pygame gym[toy_text]'")

# Main Execution
if __name__ == "__main__":
    # Create Taxi Environment with render_mode='human'
    env = gym.make("Taxi-v3", render_mode="human")

    # Environment Parameters
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    # Hyperparameters
    alpha = 0.7        # Learning Rate
    gamma = 0.618      # Discount Factor
    epsilon = 1.0      # Exploration Rate
    max_epsilon = 1.0  # Max Exploration Rate
    min_epsilon = 0.01 # Min Exploration Rate
    decay_rate = 0.001 # Adjusted Exponential Decay Rate for smoother decay

    # Training Parameters
    episodes = 10
    max_steps = 10

    # Initialize Q-Table
    Q_table = initialize_q_table(state_space_size, action_space_size)

    # Train the Agent
    Q_table, epsilon = train_agent(env, Q_table, alpha, gamma, epsilon, decay_rate, min_epsilon, episodes, max_steps)
    print("Training finished.")

    # Test the Agent
    avg_timesteps, avg_penalties = test_agent(env, Q_table, 10)
    print(f"Results after 10 episodes:")
    print(f"Average Timesteps per Episode: {avg_timesteps}")
    print(f"Average Penalties per Episode: {avg_penalties}")

    # Visualize the Taxi's Path
    visualize_agent(env, Q_table)

    env.close()
