import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from bradleyenv import BradleyAirportEnv
from cnn import MultiPlaneDQNAgent

# Hyperparameters
num_planes = 5
num_actions = 13
num_episodes = 5000
batch_size = 32
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9995
target_update_freq = 1000  # steps
max_steps_per_episode = 10000

# Initialize environment and agent
env = BradleyAirportEnv()
agent = MultiPlaneDQNAgent(num_planes, num_actions)

# Setup tracking
episode_rewards = []
losses = []

epsilon = epsilon_start
steps_done = 0

for episode in range(num_episodes):
    state, _, _ = env.reset()
    done = False
    total_reward = 0
    step_in_episode = 0

    # Initial state grid
    state = env.generate_state_grid()

    while not done and step_in_episode < max_steps_per_episode:
        # Make sure enough planes are in the environment
        while len(env.planes) < num_planes:
            env.add_plane()

        # Select actions
        actions = agent.select_actions(state, epsilon)

        # Step environment safely
        env.step(actions)

        # Observe next state
        next_state = env.generate_state_grid()

        # Reward can be summed from planes
        reward = 0
        for plane in env.planes:
            reward += 1  # +1 reward per alive plane, you can customize

        # Check if episode is done
        done = all(plane.flight_state == 3 for plane in env.planes)  # All planes at gate
        # Or done if no planes
        if len(env.planes) == 0:
            done = True

        # Store experience
        agent.store_transition(state, actions, reward, next_state, done)

        # Train agent
        agent.train(batch_size)

        # Track loss
        if hasattr(agent, 'loss_value'):
            losses.append(agent.loss_value)

        # Update for next step
        state = next_state
        total_reward += reward
        steps_done += 1
        step_in_episode += 1

        # Update target network
        if steps_done % target_update_freq == 0:
            agent.update_target_model()

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    episode_rewards.append(total_reward)

    # Plot rewards and losses every 10 episodes
    if episode % 10 == 0:
        clear_output(wait=True)
        plt.figure(figsize=(12,5))
        
        plt.subplot(1,2,1)
        plt.title('Rewards')
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.subplot(1,2,2)
        plt.title('Loss')
        plt.plot(losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')

        plt.show()

    print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

# Save model at the end
agent.save("dqn_airport_final.pth")
print("Training finished and model saved!")