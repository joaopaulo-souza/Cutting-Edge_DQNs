import gymnasium as gym
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DQN import DQN

print("Libraries Imported") 
#======================================================
# Control Panel
#======================================================

# Episodes
human_render = False
max_episodes = 5000
episode_solution = 300
episode_time_steps = 300
human_render_after_episodes = 500

# DQN Parameter Dictonary
DQN_Parameters = {
    "discount_factor": 4,
    "learning_rate": 0.05,
    "epsilon_max": 1,
    "epsilon_min": 0.05,
    "epsilon_max_exploration_steps": 100000.0,
    "epsilon_random_frames": 5000,

    "update_policy_net_steps": 4,
    "update_target_net_steps": 10,

    "experience_replay_length": 100000,
    "batch_size": 128,

    "double_dqn_flag": True,
    "target_net_flag": True,
    }
dqn = DQN(DQN_Parameters)

#Initialize Enviroment
game_name = "LunarLander-v2"
env = gym.make(game_name)
observation_space = env.observation_space
action_space = env.action_space
status_shape = observation_space.shape
N_actions = action_space.n
N_inputs = observation_space.shape[0]

# Neural Net Parameters dictionary
NN_Parameters = {
    "n_inputs": N_inputs,
    "n_hidden_layer_1": 20,
    "n_hidden_layer_2": 10,
    "n_hidden_layer_3": 0,
    "n_outputs": N_actions,
    "activation_function": "relu", # relu, tanh, sigmoid, leaky_relu  
    "loss_function": nn.SmoothL1Loss(),
    }

# Create NNs
dqn.SetNNParameters(NN_Parameters)
dqn.CreateQnets()

#Counters and flags
episode_reward = 0
episode_reward_history = []
episode_count = 0
frame_count = 0

#=================================================================
#   Main Loop
#=================================================================


for ep in range(max_episodes):
    pdb.set_trace()
    #Initial observation of the episode
    obs = env.reset()
    state = np.array(obs[0])
    episode_reward = 0
    for t in range(episode_time_steps):
        frame_count += 1
        action = np.random.choice(N_actions)
        

        #Apply the action in enviroment
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state)

        #Compute episode reward
        episode_reward += reward

        #Experience tuple 
        exp_tuple = (state,action,reward,next_state,done)

        #Update state
        state = next_state
        
        if frame_count % update_after_actions == 0 and len(Exp_replay) > batch_size:
            a = 0            
                
    # Update running reward to check condition for solving
    episode_count += 1
    print("Episode {} Reward: {:.3f}".format(episode_count, episode_reward))
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = episode_reward_history[-1]


    if running_reward > episode_solution:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

    if (episode_count >= max_episodes): # Maximum number of episodes reached
        print("Stopped at episode {}!".format(episode_count))
        break
    # See one episode in render mode
    if human_render == True:
        human_rendering = False
        env.close()
        env = gym.make(game_name)

    # Turn off human-rendering 
    if (episode_count % human_render_after_episodes == 0):
        human_render = True
        env.close()
        env = gym.make(game_name, render_mode="human")
            

        
#End Enviroment
env.close()


