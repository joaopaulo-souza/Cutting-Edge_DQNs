import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        self.loss_function = keras.losses.MeanSquaredError()


    def CreateNN(self):

        #Number of hidden layers   
        if self.n_hidden_layer_3 > 0:
            model = keras.Sequential()
            model.add(layers.Dense(self.n_hidden_layer_1, activation = self.activation_function))
            model.add(layers.Dense(self.n_hidden_layer_2, activation = self.activation_function))
            model.add(layers.Dense(self.n_hidden_layer_3, activation = self.activation_function))
            model.add(layers.Dense(self.n_outputs, activation = None))
            model.add(layers.Softmax())
            
        else:
            model = keras.Sequential()
            model.add(layers.Dense(self.n_hidden_layer_1, activation = self.activation_function))
            model.add(layers.Dense(self.n_hidden_layer_2, activation = self.activation_function))
            model.add(layers.Dense(self.n_outputs, activation = None))
            model.add(layers.Softmax())
        return model

    def CreateQnets(self):
        self.policy_net = self.CreateNN()
        self.target_net = self.CreateNN()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate,clipnorm=1.0)

    def GetEpsilon(self):
        return self.epsilon

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
            state_tensor = tf.convert_to_tensor(state)
            #Convert to matrix of 1 colum
            state_tensor = tf.expand_dims(state_tensor,0)
            #Calculate the Q-values
            Q_actions = self.policy_net(state_tensor, training = False)
            #Take the best action
            action = tf.math.argmax(Q_actions[0]).numpy()
        return action

    def AddToReplay(self, experience):
        self.experience_replay.append(experience)
        if len(self.experience_replay) > self.experience_replay_length:
            del self.experience_replay[:1]

    def UpdatePolicyNet(self,frame_count):
        if frame_count % self.update_policy_net_steps ==0 and len(self.experience_replay) > self.batch_size:
            indices = np.random.choice(range(len(self.experience_replay)), size = self.batch_size)
            exp_batch = [self.experience_replay[i] for i in indices]

            state_sample = tf.stack([tf.convert_to_tensor(exp[0]) for exp in exp_batch])
            action_sample = tf.convert_to_tensor([exp[1] for exp in exp_batch])
            rewards_sample = tf.convert_to_tensor([exp[2] for exp in exp_batch])
            next_state_sample = tf.stack([tf.convert_to_tensor(exp[3]) for exp in exp_batch])
            done_sample = tf.convert_to_tensor([float(exp[4]) for exp in exp_batch])

            if self.no_target_net_flag:
                #No Target DQN
                future_Q_values = self.policy_net(next_state_sample)
                updated_Q_values = rewards_sample + self.discount_factor * tf.reduce_max(future_Q_values, axis=1)
            
            elif self.double_dqn_flag: 
                #Double DQN
                future_Q_values = self.policy_net(next_state_sample)
                future_action = tf.math.argmax(future_Q_values, axis = 1)
                future_Q_target_values = self.target_net(next_state_sample)
                masks = tf.one_hot(future_action,N_actions)
                future_Q_target_actions = tf.math.reduce_sum((future_Q_target_values * masks), axis=1)
                    
                updated_Q_values = rewards_sample + gamma * future_Q_target_actions

            else:
                #Simple DQN
                future_Q_values = self.target_net(next_state_sample)
                updated_Q_values = rewards_sample + self.discount_factor * tf.reduce_max(future_Q_values, axis = 1)

            #Done samples ( If the state is terminal, set to -1)
            updated_Q_values = updated_Q_values * (1 - done_sample) - done_sample
            #Create a mask so we can calculate loss ( will do One Hot Encoding in actions)
            masks = tf.one_hot(action_sample, self.n_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                Q_values = self.policy_net(state_sample)
                        
                # Apply the masks to the Q-values to get the Q-value for action taken
                Q_action = tf.math.reduce_sum((Q_values * masks), axis=1)
                        
                # Calculate loss between new Q-value (updated usnig Bell's eq.) and old Q-value (policy_net)
                loss = self.loss_function(updated_Q_values, Q_action)

            #Back Propagation
            grads = tape.gradient(loss, self.policy_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
            

    def UpdateTargetNet(self, frame_count):
        if frame_count % self.update_target_net_steps == 0:
            self.target_net.set_weights(self.policy_net.get_weights())
        


        
        
            
        
        
        
        
        
        
        
