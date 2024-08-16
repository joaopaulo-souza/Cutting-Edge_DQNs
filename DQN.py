import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pdb

class DQN:
    def __init__(self, parameters_dictionary):
        D = parameters_dictionary
        #DQN e-Greedy
        self.discount_factor = D["discount_factor"]
        self.learning_rate = D["learning_rate"]
        self.epsilon_max = D["epsilon_max"]
        self.epsilon_min = D["epsilon_min"]
        self.epsilon = self.epsilon_max
        self.epsilon_max_exploration_steps = D["epsilon_max_exploration_steps"]
        self.epsilon_decay = (self.epsilon_max - self.epsilon_min) / self.epsilon_max_exploration_steps
        self.epsilon_random_frames = D["epsilon_random_frames"]
        #DQN network
        self.policy_net = None
        self.target_net = None
        self.network_hidden_layers_dim = None
        self.n_layers = None
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
        self.n_inputs = D["n_inputs"]
        self.n_hidden_layer_1 = D["n_hidden_layer_1"]
        self.n_hidden_layer_2 = D["n_hidden_layer_2"]
        self.n_hidden_layer_3 = D["n_hidden_layer_3"]
        self.n_outputs = D["n_outputs"]
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
                nn.Softmax()
                )
        else:
            model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_hidden_layer_1),
                activation,
                nn.Linear(self.n_hidden_layer_1, self.n_hidden_layer_2),
                activation,
                nn.Linear(self.n_hidden_layer_2, self.n_outputs),
                nn.Softmax()
                )
        return model

    def CreateQnets(self):
        self.policy_net = self.CreateNN()
        self.target_net = self.CreateNN()
        #pdb.set_trace()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=self.learning_rate)

    def EpsilonCalc(self):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon,self.epsilon_min)
        return self.epsilon

    def RandomAction(self):
        action = np.random.choice(self.action_space)
        return action

    def Q_Action(state):
        #Convert state to tensor
        state_tensor = None
        #Convert to matrix of 1 colum
        state_tensor = None
        #Calculate the Q-values
        Q_actions = self.network(state_tensor)
        #Take the best action
        action = None

    def AddToReplay(self, experience):
        self.replay_experience.append(experience)
        if len(self.experience_replay) > self.experience_replay_length:
            del Exp_replay[:1]

    def UpdatePolicyNet(self):
        indices = np.random.choice(range(len(self.experience_replay)), size = self.batch_size)
        exp_batch = [self.experience_replay[i] for i in indices]

        state_sample = np.array([exp_replay[i][0] for i in exp_batch])
        action_sample = [exp_replay[i][1] for i in exp_batch]
        rewards_sample = [exp_replay[i][2] for i in exp_batch]
        next_state_sample = np.array([exp_replay[i][3] for i in exp_batch])
        done_sample = tf.convert_to_tensor([float(exp_replay[i][4]) for i in exp_batch])
        
        if self.double_dqn == True:
            #Double DQN
            future_Q_values = DQN(next_state_sample)
            future_action = tf.math.argmax(future_Q_values, axis = 1)
            future_Q_target_values = DQN_T(next_state_sample)
            masks = tf.one_hot(future_action,N_actions)
            future_Q_target_actions = tf.math.reduce_sum((future_Q_target_values * masks), axis=1)
                
            updated_Q_values = rewards_sample + gamma * future_Q_target_actions

        else:
            #Simple DQN
            future_Q_values = DQN_T(next_state_sample)
            updated_Q_values = rewards_sample + gamma * tf.reduce_max(future_Q_values, axis = 1)

        #Done samples
        updated_Q_values = updated_Q_values * (1 - done_sample) - done_sample
        #Create a mask so we can calculate loss
        masks = tf.one_hot(action_sample,N_actions)

        # Train the model on the states and updated Q-values
        Q_values = DQN(state_sample)
                    
        # Apply the masks to the Q-values to get the Q-value for action taken
        Q_action = tf.math.reduce_sum((Q_values * masks), axis=1)
                    
        # Calculate loss between new Q-value and old Q-value
        loss = self.loss_function(updated_Q_values, Q_action)

        #Back Propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def UpdateTargetNet(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        


        
        
            
        
        
        
        
        
        
        
