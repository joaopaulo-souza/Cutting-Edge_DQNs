import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
import pdb

class ReplayBuffer():
    def __init__(self, max_size=20000, batch_size=64):
        self.max_size = max_size
        self.mem_size = 0
        self.batch_size = batch_size
        self.replay = []
        
    def store(self, experience):
        self.replay.append(experience)
        if len(self.replay) > self.max_size:
            del self.replay[:1]

    def sample(self):
        indices = np.random.choice(range(len(self.replay)), size = self.batch_size)
        experience_batch = [self.replay[i] for i in indices]

        states, actions, rewards, states_, dones = [],[],[],[],[]

        for i in range(self.batch_size):
            states.append(experience_batch[i][0])
            actions.append(experience_batch[i][1])
            rewards.append(experience_batch[i][2])
            states_.append(experience_batch[i][3])
            dones.append(float(experience_batch[i][4]))

        return states, actions, rewards, states_, dones
        
        
class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, dueling_dqn_flag):
        super(DeepQNetwork, self).__init__()

        self.dueling_dqn_flag = dueling_dqn_flag

        self.layer1 = nn.Linear(*input_dims,256)
        self.layer2 = nn.Linear(256,128)
        self.layer3 = nn.Linear(128,n_actions)

        self.ValueLayer = nn.Linear(128,1)
        self.AdvantageLayer = nn.Linear(128,n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(y))
        if not self.dueling_dqn_flag:
            output = self.layer3(y)
            return output
        if self.dueling_dqn_flag:
            V = self.ValueLayer(y)
            A = self.AdvantageLayer(y)
            return V, A

class Agent():
    def __init__(self, input_dims, n_actions, lr=1e-6, gamma=0.99,
                 epsilon=1.0, epsilon_dec = 1e-5, epsilon_min=0.01,
                 replay_size = 20000, batch_size = 64, frames_per_update = 4,
                 update_target_frames =1000, double_dqn_flag=True,
                 dueling_dqn_flag=True):

        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.frames_per_update = frames_per_update
        self.update_target_frames = update_target_frames

        self.frame_count = 0

        self.double_dqn_flag = double_dqn_flag
        self.dueling_dqn_flag = dueling_dqn_flag

        self.action_space = [i for i in range(self.n_actions)]
        self.policy_net = DeepQNetwork(self.lr, self.n_actions, self.input_dims, self.dueling_dqn_flag)
        self.target_net = DeepQNetwork(self.lr, self.n_actions, self.input_dims, self.dueling_dqn_flag)
        self.exp_replay = ReplayBuffer(self.replay_size, self.batch_size)

    def IncrementFrame(self):
        self.frame_count += 1 
        
    def Policy(self, state):
        if np.random.uniform() > self.epsilon:
            state = T.tensor(state, dtype=T.float).to(self.policy_net.device)
            if not self.dueling_dqn_flag:
                action = T.argmax(self.policy_net(state)).item()
            if self.dueling_dqn_flag:
                V, A = self.policy_net.forward(state)
                Q = V + (A - A.mean())
                action = T.argmax(Q).item()
                # Since V and A.mean are constants, they make no difference
                #action = T.argmax(A).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def DecrementEpsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def ReplayStore(self, experience):
        self.exp_replay.store(experience)

    def SampleReplay(self):
        #Get the samples
        states, actions, rewards, states_, dones = self.exp_replay.sample()

        #Transform samples to tensors
        states = T.tensor(np.stack(states), dtype = T.float).to(self.policy_net.device)
        actions = T.tensor(np.stack(actions)).to(self.policy_net.device)
        rewards = T.tensor(np.stack(rewards), dtype = T.float).to(self.policy_net.device)
        states_ = T.tensor(np.stack(states_), dtype = T.float).to(self.policy_net.device)
        dones = T.tensor(np.stack(dones), dtype = T.float).to(self.policy_net.device)

        return states, actions, rewards, states_, dones

    def Learn(self):
        if self.frame_count % self.frames_per_update ==0 and len(self.exp_replay.replay) > self.batch_size:
            #Zero gradients
            self.policy_net.optimizer.zero_grad()

            #Update target network
            self.UpdateTargetNet()

            #Get the samples
            states, actions, rewards, states_, dones = self.SampleReplay()

            #Indices for actions
            indices = np.arange(self.batch_size)

            #Not-Dueling dqn
            if not self.dueling_dqn_flag:
                #Get Q_p(s,a)
                Q_pred = self.policy_net.forward(states)[indices, actions]
                #get Q_p(s',a')
                Q_p_next = self.policy_net.forward(states_)
                #Get Q_t(s',a')
                Q_t_next = self.target_net.forward(states_)
            #Dueling dqn
            if self.dueling_dqn_flag:
                V_pred, A_pred = self.policy_net.forward(states)
                V_p_next, A_p_next = self.policy_net.forward(states_)
                V_t_next, A_t_next = self.target_net.forward(states_)
                #Get Q_p(s,a)
                Q_pred = T.add(V_pred, (A_pred - A_pred.mean(dim=1, keepdim=True)))[indices, actions]
                #get Q_p(s',a')
                Q_p_next = T.add(V_p_next, (A_p_next - A_p_next.mean(dim=1, keepdim=True)))
                #Get Q_t(s',a')
                Q_t_next = T.add(V_t_next, (A_t_next - A_t_next.mean(dim=1,keepdim=True)))

            

            #Not-Double dqn
            if not self.double_dqn_flag:
                #Apply Bellman's eq. 
                Q_target = rewards + self.gamma * Q_t_next.max(dim=1)[0] * (1.0 - dones)
            #Double dqn
            if self.double_dqn_flag:
                #Get argmax(Q_p(s',a))
                argmax_actions = T.argmax(Q_p_next, dim = 1)
                #Get Q_t(s',argmax(Q_p(s',a)))
                Q_t_next = Q_t_next[indices,argmax_actions]
                #Apply Bellman's eq.
                Q_target = rewards + self.gamma * Q_t_next * (1.0 - dones)
                
            
            #Calculate Loss
            loss = self.policy_net.loss(Q_target, Q_pred).to(self.policy_net.device)
            loss.backward()
            self.policy_net.optimizer.step()

            self.DecrementEpsilon()

    def GetEpsilon(self):
        return self.epsilon

    def UpdateTargetNet(self):
        if self.frame_count % self.update_target_frames == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        
        
        

        
        
            
            
            
            
        
        
        

        

        

        
