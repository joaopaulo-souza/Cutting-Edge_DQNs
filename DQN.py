import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pdb

class DQN:
    def __init__(self, parameters_dictionary):
        D = parameters_dictionary
        #DQN e-Greedy
        self.discount_factor = D["discount_factor"]
        self.learning_rate = D["learning_rate"]
        self.action_space = D["action_space"]
        self.n_actions = D["n_actions"]
        self.action_space = D["action_space"]
        self.state_shape = D["state_shape"]
        self.epsilon_max = D["epsilon_max"]
        self.epsilon_min = D["epsilon_min"]
        self.epsilon = self.epsilon_max
        self.epsilon_max_exploration_steps = D["epsilon_max_exploration_steps"]
        self.epsilon_decay = (self.epsilon_max - self.epsilon_min) / self.epsilon_max_exploration_steps
        self.epsilon_random_frames = D["epsilon_random_frames"]
        #DQN network
        self.policy_net = None
        self.target_net = None
        self.n_inputs = self.state_shape
        self.n_hidden_layer_1 = None
        self.n_hidden_layer_2 = None
        self.n_hidden_layer_3 = None
        self.n_outputs = self.n_actions
        self.activation_function = None 
        self.loss_function = None
        self.optimizer = None
        self.update_policy_net_steps = D["update_policy_net_steps"]
        self.update_target_net_steps = D["update_target_net_steps"]
        self.double_dqn_flag = D["double_dqn_flag"]
        #Replay Experience
        self.experience_replay = []
        self.experience_replay_length = D["experience_replay_length"]
        self.batch_size = D["batch_size"]

    def SetNNParameters(self, parameters_dictionary):
        D = parameters_dictionary
        self.n_hidden_layer_1 = D["n_hidden_layer_1"]
        self.n_hidden_layer_2 = D["n_hidden_layer_2"]
        self.n_hidden_layer_3 = D["n_hidden_layer_3"]
        self.activation_function = D["activation_function"]
        self.loss_function = D["loss_function"]


    def CreateNN(self):
        #Activation Function
        if self.activation_function == "tanh":
            activation = nn.Tanh()
        if self.activation_function == "relu":
            activation = nn.ReLU()
        if self.activation_function == "sigmoid":
            activation = nn.Sigmoid()
        if self.activation_function == "leaky_relu":
            activation = nn.LeakyReLU()

        #Number of hidden layers   
        if self.n_hidden_layer_3 > 0:
            model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_hidden_layer_1),
                activation,
                nn.Linear(self.n_hidden_layer_1, self.n_hidden_layer_2),
                activation,
                nn.Linear(self.n_hidden_layer_2, self.n_hidden_layer_3),
                activation,
                nn.Linear(self.n_hidden_layer_3, self.n_outputs),
                nn.Softmax(dim=1)
                )
        else:
            model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_hidden_layer_1),
                activation,
                nn.Linear(self.n_hidden_layer_1, self.n_hidden_layer_2),
                activation,
                nn.Linear(self.n_hidden_layer_2, self.n_outputs),
                nn.Softmax(dim=1)
                )
        return model

    def CreateQnets(self):
        self.policy_net = self.CreateNN()
        self.target_net = self.CreateNN()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=self.learning_rate)

    def E_GreedyPolicy(self, state):
        #Compute epsilon
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon,self.epsilon_min)
        #Calculate de probability of exploration
        p_exploration = np.random.uniform()
        if p_exploration < self.epsilon:
            #take random action
            action = np.random.choice(self.action_space)
        else:
            #take calculated action
            #Convert state to tensor
            state_tensor = torch.tensor(state)
            #Convert to matrix of 1 colum
            state_tensor = state_tensor.unsqueeze(0)
            #Calculate the Q-values
            Q_actions = self.network(state_tensor)
            #Take the best action
            action = torch.argmax(Q_actions).item()
        return action

    def AddToReplay(self, experience):
        self.experience_replay.append(experience)
        if len(self.experience_replay) > self.experience_replay_length:
            del self.experience_replay[:1]

    def UpdatePolicyNet(self,frame_count):
        if frame_count % self.update_policy_net_steps ==0 and len(self.experience_replay) > self.batch_size:
            pdb.set_trace()
            indices = np.random.choice(range(len(self.experience_replay)), size = self.batch_size)
            exp_batch = [self.experience_replay[i] for i in indices]

            state_sample = torch.stack([torch.tensor(exp[0]) for exp in exp_batch])
            action_sample = torch.tensor([exp[1] for exp in exp_batch])
            rewards_sample = torch.tensor([exp[2] for exp in exp_batch])
            next_state_sample = torch.stack([torch.tensor(exp[3]) for exp in exp_batch])
            done_sample = torch.tensor([float(exp[4]) for exp in exp_batch])
            
            if self.double_dqn_flag == True: 
                #Double DQN
                future_Q_values = self.policy_net(next_state_sample)
                future_action = tf.math.argmax(future_Q_values, axis = 1)
                future_Q_target_values = DQN_T(next_state_sample)
                masks = tf.one_hot(future_action,N_actions)
                future_Q_target_actions = tf.math.reduce_sum((future_Q_target_values * masks), axis=1)
                    
                updated_Q_values = rewards_sample + gamma * future_Q_target_actions

            else:
                #Simple DQN
                future_Q_values = self.target_net(next_state_sample)
                updated_Q_values = rewards_sample + self.discount_factor * torch.max(future_Q_values,dim=1)[0]

            #Done samples ( If the state is terminal, set to -1)
            updated_Q_values = updated_Q_values * (1 - done_sample) - done_sample
            #Create a mask so we can calculate loss ( will do One Hot Encoding in actions)
            masks = F.one_hot(action_sample, self.n_actions)

            # Train the model on the states and updated Q-values
            Q_values = self.policy_net(state_sample)
                        
            # Apply the masks to the Q-values to get the Q-value for action taken
            Q_action = torch.sum((Q_values * masks), dim=1)
                        
            # Calculate loss between new Q-value (updated usnig Bell's eq.) and old Q-value (policy_net)
            loss = self.loss_function(updated_Q_values, Q_action)

            #Back Propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

    def UpdateTargetNet(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        


        
        
            
        
        
        
        
        
        
        
