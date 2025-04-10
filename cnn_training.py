import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from bradleyenv import BradleyAirportEnv
from cnn import MultiPlaneDQNAgent
import random
import time

# Hyperparameters
num_planes = 5
num_actions = 13
num_episodes = 5000
batch_size = 32
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9995
target_update_freq = 500  # steps
max_steps_per_episode = 10000

# Rendering parameters
RENDER = False            # Toggle rendering ON or OFF
RENDER_EVERY_N_EPISODES = 50
SLEEP_TIME = 0.01         # Sleep 10ms to make rendering smoother

# Initialize environment and agent
env = BradleyAirportEnv()
agent = MultiPlaneDQNAgent(num_planes, num_actions)

# state, _, _ = env.reset()
# state = env.generate_state_grid()

# while len(agent.memory) < batch_size:
#     # Add new planes at random
#     if random.random() < 0.05:
#         env.add_plane()

#     # Fill memory with random actions
#     actions = agent.select_actions(state, epsilon=1.0)  # Take completely random actions
#     next_state, reward, done = env.step(actions)
#     agent.store_transition(state, actions, reward, next_state, done)
#     state = next_state

#     if done:
#         state, _, _ = env.reset()
#         state = env.generate_state_grid()


# Setup tracking
episode_rewards = []
losses = []

epsilon = epsilon_start
steps_done = 0

# plt.ion()  # turn interactive mode ON
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

for episode in range(num_episodes):
    state, _, _ = env.reset()
    done = False
    total_reward = 0
    steps_in_episode = 0

    # Initial state grid
    state = env.generate_state_grid()

    while not done and steps_in_episode < max_steps_per_episode:
        # Add new planes at random
        if len(env.planes) < num_planes:
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

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    episode_rewards.append(total_reward)

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

    print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward:.2f} | Steps = {steps_in_episode} | Epsilon: {epsilon:.3f}")

# Save model at the end
agent.save("dqn_airport_final.pth")
print("Training finished and model saved!")