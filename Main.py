import gymnasium as gym
import pdb
import numpy as np
import time 


print("Libraries Imported") 
#======================================================
# Control Panel
#======================================================

#Initialize Enviroment
game_name = "LunarLander-v2"
print("Game enviroment: {}".format(game_name))
env = gym.make(game_name)
observation_space = env.observation_space 
n_actions = env.action_space.n  #Use only for discrete action space
action_space = tuple([i for i in range(n_actions)])
status_shape = observation_space.shape

# Framework (torch, tensorflow)
framework = "torch"

if framework == "torch":
    from DQN_torch import DQN
if framework == "tensorflow":
    from DQN_tensorflow import DQN

print("Framework used: {}".format(framework))


# Episodes
human_render = False
max_episodes = 5000
episode_solution = 300
episode_time_steps = 300
human_render_after_episodes = 500

# DQN Parameter Dictonary
DQN_Parameters = {
    "discount_factor": 0.95,
    "learning_rate": 0.05,
    "n_actions": n_actions,
    "action_space": action_space,
    "state_shape": observation_space.shape[0],
    "epsilon_max": 1,
    "epsilon_min": 0.05,
    "epsilon_max_exploration_steps": 100000.0,
    "epsilon_random_frames": 5000,

    "update_policy_net_steps": 4,
    "update_target_net_steps": 500,

    "experience_replay_length": 100000,
    "batch_size": 512,

    "double_dqn_flag": False,
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
    "n_hidden_layer_1": 30,
    "n_hidden_layer_2": 30,
    "n_hidden_layer_3": 0,
    "activation_function": "tanh", # relu, tanh, sigmoid, leaky_relu
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
    
    #Initial observation of the episode
    obs = env.reset()
    
    #Convert to 32bits 
    state = np.float32(obs[0])

    #Zero the total episode reward
    episode_reward = 0

    #Measure episode time
    start_time = time.time()
    for t in range(episode_time_steps):
        frame_count += 1

        #Take action based on E-Greedy Policy
        action = dqn.E_GreedyPolicy(state)

        #Apply the action in enviroment
        next_state, reward, done, _, _ = env.step(action)

        #Convert to 32bits
        next_state = np.float32(next_state)
        reward = np.float32(reward)
        
        #Compute episode reward
        episode_reward += reward

        #Experience tuple 
        experience = (state,action,reward,next_state,done)

        #Add experience to Experience Replay
        dqn.AddToReplay(experience)

        #Update state
        state = next_state

        #Update Policy Net
        dqn.UpdatePolicyNet(frame_count)

        #Update Target Net
        dqn.UpdateTargetNet(frame_count)

        if done:
            break

    #Measure episode end time
    end_time = time.time()
    
    # Update running reward to check condition for solving
    episode_count += 1

    # Log details
    epsilon = dqn.GetEpsilon()
    episode_time = end_time - start_time
    line_temp = "running reward: {:.2f} at episode {}, frame count {}, epsilon {:.3f}, episode_time {:.3f}s"
    print(line_temp.format(episode_reward, episode_count, frame_count, epsilon, episode_time))
    
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]

    if episode_reward > episode_solution:  # Condition to consider the task solved
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


