import gymnasium as gym
import pdb
import numpy as np
import time 
from DQN import Agent
from Utils import plot_learning_curve

print("Libraries Imported") 
#======================================================
# Control Panel
#======================================================

#Initialize Enviroment
game_name = "LunarLander-v2"
#game_name = "CartPole-v1"
print("Game enviroment: {}".format(game_name))
env = gym.make(game_name)
observation_space = env.observation_space 
n_actions = env.action_space.n  
state_dim = observation_space.shape

# Episodes
human_render = False
max_episodes = 15000
episode_solution = 300
episode_time_steps = 300
human_render_flag = False
stop_flag = False
human_render_after_episodes = 500
print_after_episodes = 100

double_dqn_flag = True
dueling_dqn_flag = True

# DQN Parameters
gamma = 0.99
lr = 1e-4
epsilon_dec = 1e-5

# Object Creation
dqn_agent = Agent(state_dim, n_actions, lr, gamma, epsilon_dec = epsilon_dec)


#Counters and flags
episode_score = 0
scores = []
epsilon_history = []
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
    episode_score = 0

    #Measure episode time
    start_time = time.time()

    #while True:
    for t in range(episode_time_steps):
        #frame_count +1
        dqn_agent.IncrementFrame()

        #Take action based on E-Greedy Policy
        action = dqn_agent.Policy(state)

        #Apply the action in enviroment
        next_state, reward, done, _, _ = env.step(action)

        #Convert to 32bits
        next_state = np.float32(next_state)
        reward = np.float32(reward)
        
        #Compute episode reward
        episode_score += reward

        #Experience tuple 
        experience = (state,action,reward,next_state,done)

        #Add experience to Experience Replay
        dqn_agent.ReplayStore(experience)

        #Update state
        state = next_state

        #Update Policy Net
        dqn_agent.Learn()

        if done:
            break

    #Measure episode end time
    end_time = time.time()
    
    # Update running reward to check condition for solving
    episode_count += 1

    # Log details
    current_epsilon = dqn_agent.GetEpsilon()
    epsilon_history.append(current_epsilon)
    episode_time = end_time - start_time
    scores.append(episode_score)
    avg_score = np.mean(scores[-100:])

    if ep % print_after_episodes == 0:
        line_temp = "score: {:.2f} at episode {}, average score{:.2f}, epsilon {:.3f}, episode_time {:.3f}s"
        print(line_temp.format(episode_score, episode_count, avg_score, current_epsilon, episode_time))
    
    # Stopping Conditionals
    if episode_score > episode_solution and stop_flag == True:  
        print("Solved at episode {}!".format(episode_count))
        break

    if episode_count >= max_episodes: # Maximum number of episodes reached
        print("Stopped at episode {}!".format(episode_count))
        break

    # See one episode in render mode
    if human_render == True:
        human_render = False
        env.close()
        env = gym.make(game_name)

    # Turn off human-rendering 
    if (episode_count % human_render_after_episodes == 0) and human_render_flag == True:
        human_render = True
        env.close()
        env = gym.make(game_name, render_mode="human")
            

        
#End Enviroment
env.close()
#Plot learning curve

#Dueling 
if dueling_dqn_flag:
    dueling = "Dueling-"
else:
    dueling = ""

#Double
if double_dqn_flag:
    double = "Double-"
else:
    double = ""

#Algorithm name
algo = dueling + double

#File name
filename = game_name + algo + "dqn.png"
x = [i+1 for i in range(max_episodes)]
plot_learning_curve(x, scores, epsilon_history, filename)

