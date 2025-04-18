import numpy as np
import matplotlib.pyplot as plt
from bradleyenv import BradleyAirportEnv
from cnn import MultiPlaneDQNAgent
import time
import torch

# Hyperparameters
num_planes = 3
num_actions = 12
num_episodes = 1000
batch_size = 32
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9975
target_update_freq = 300  # steps
max_steps_per_episode = 5000

# Rendering parameters
RENDER = False            # Toggle rendering ON or OFF
RENDER_EVERY_N_EPISODES = 50
SLEEP_TIME = 0.01         # Sleep 10ms to make rendering smoother

# Initialize environment and agent
env = BradleyAirportEnv()
agent = MultiPlaneDQNAgent(num_planes, num_actions)
print(next(agent.model.parameters()).device)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))  # Print GPU name
    print(f"Memory Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 2)} GB")
    print(f"Memory Cached: {round(torch.cuda.memory_reserved(0)/1024**3, 2)} GB")
else:
    print("Warning: Running on CPU! Training will be much slower.")


# Setup tracking
episode_rewards = []
losses = []

epsilon = epsilon_start
steps_done = 0

# plt.ion()  # turn interactive mode ON
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

training_start_time = time.time()

for episode in range(num_episodes):
    start_time = time.time()
    state, _, _ = env.reset()
    done = False
    total_reward = 0
    steps_in_episode = 0

    # Initial state grid
    state = env.generate_state_grid()

    while not done and steps_in_episode < max_steps_per_episode:

        # Make sure enough planes are present
        while len(env.planes) < num_planes:
            env.add_plane()

        # Select actions
        actions = agent.select_actions(state, epsilon)

        # Step environment safely
        next_state, step_reward_sum, per_plane_rewards, done = env.step(actions)

        if RENDER and episode % RENDER_EVERY_N_EPISODES == 0:
            env.render()
            time.sleep(SLEEP_TIME)  # Small pause to see movement
        
        # Store experience
        agent.store_transition(state, actions, per_plane_rewards, next_state, done)

        # Train agent
        if steps_done % 4 == 0: # to speed up the training
            agent.train(batch_size)

        # Track loss
        if hasattr(agent, 'loss_value'):
            losses.append(agent.loss_value)

        # Update for next step
        state = next_state
        total_reward += sum(per_plane_rewards)
        steps_done += 1
        steps_in_episode += 1

        # Update target network
        if steps_done % target_update_freq == 0:
            agent.update_target_model()

    progress = episode / num_episodes  # progress from 0.0 to 1.0
    env.set_training_progress(progress)

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    episode_rewards.append(total_reward)
    k = min(1000, len(losses))
    if len(losses) > 0:
      avg_loss = np.mean(losses[-k:])
      print(f"Average Loss: {avg_loss:.6f}")

    # # Plot rewards and losses every 10 episodes
    # if episode % 10 == 0:
    #     ax1.clear()
    #     ax2.clear()
        
    #     ax1.set_title('Rewards')
    #     ax1.plot(episode_rewards)
    #     ax1.set_xlabel('Episode')
    #     ax1.set_ylabel('Total Reward')

    #     ax2.set_title('Loss')
    #     ax2.plot(losses)
    #     ax2.set_xlabel('Training Step')
    #     ax2.set_ylabel('Loss')

    #     plt.pause(0.001)  # short pause so matplotlib refreshes

    elapsed_time = time.time() - start_time
    print(f"Episode {episode+1}/{num_episodes} completed in {elapsed_time:.2f} seconds")

    print(f"Total Reward: {total_reward:.2f} | Steps = {steps_in_episode} | Epsilon: {epsilon:.3f}")

training_elapsed_time = time.time() - training_start_time
print(f"Full Training Completed in {training_elapsed_time/60:.2f} minutes")

# Save model at the end
agent.save("dqn_airport_final2.pth")
print("Training finished and model saved!")

# Plot Q-value Diagnostic
if hasattr(agent, 'max_q_list'):
    plt.figure(figsize=(10, 6))
    plt.plot(agent.max_q_list, label="Max Q-value per training step")
    plt.title("Maximum Q-Value Over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Max Q-value")
    plt.legend()
    plt.grid(True)
    plt.show()

def moving_average(values, window=10):
    values = np.array(values)
    if len(values) < window:
        return values  # Don't smooth if too few points
    return np.convolve(values, np.ones(window)/window, mode='valid')


# Smooth rewards and losses
avg_rewards = moving_average(episode_rewards, window=10)
avg_losses = moving_average(losses, window=10)

# Create plots
plt.figure(figsize=(14, 6))

# Rewards Plot
plt.subplot(1, 2, 1)
plt.plot(avg_rewards)
plt.title('Smoothed Total Rewards (window=10)')
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')

# Losses Plot
plt.subplot(1, 2, 2)
plt.plot(avg_losses)
plt.title('Smoothed Loss (window=10)')
plt.xlabel('Training Step')
plt.ylabel('Average Loss')

plt.tight_layout()
plt.show()
plt.savefig('training_plots.png')