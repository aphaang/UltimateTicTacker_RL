from agents.NN import NN
import numpy as np
import torch
import copy

'''
QAgent that expects a board (81-d vector) and an encoding of legal moves (81-d vector), 
and returns the q-values of the moves on board. 
Input: 162-d vector
Output: 81-d vector  

In this case, state is a combination of the board and the valid moves available to the agent. 
'''

def one_hot(valid_actions):
    actions = np.zeros(81)
    for v in valid_actions: 
        actions[v] = 1
    return actions

class QL_agent():
    def __init__(self, env, input_dims, output_dims, gamma, epsilon, lr, batch_size, n_actions, max_mem_size=10000, 
                 eps_end=0.01, eps_dec=5e-4):
        super(QL_agent, self).__init__()
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.mem_cntr = 0
        self.counter=0
        self.nn = NN(input_dims, fc1_dims=256, fc2_dims=256, output_dims=output_dims, lr=self.lr)
        self.evalNN=copy.deepcopy(self.nn)
        self.state_memory = np.zeros((self.max_mem_size,81), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem_size, 81), dtype=np.float32)
        self.new_valid_moves_memory = np.zeros((self.max_mem_size,81))
        self.valid_moves_memory = np.zeros((self.max_mem_size,81))

        self.action_memory = np.zeros((self.max_mem_size, 81), dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.bool)

    def remember_transition(self, state, action, valid_actions, new_state, new_valid_actions, reward, done):
        index = self.mem_cntr % self.max_mem_size

        valid_actions = one_hot(valid_actions)
        action = one_hot([action])
        new_valid_actions = one_hot(new_valid_actions)

        self.state_memory[index] = copy.deepcopy(state)
        self.action_memory[index] = copy.deepcopy(action)
        self.valid_moves_memory[index] = copy.deepcopy(valid_actions)

        self.new_state_memory[index] = copy.deepcopy(new_state)
        self.new_valid_moves_memory[index] = copy.deepcopy(new_valid_actions)

        self.terminal_memory[index] = done
        self.reward_memory[index] = copy.deepcopy(reward)
        
        self.mem_cntr += 1
    
    def choose_action(self, grid, valid_actions):
        if np.random.random() > self.epsilon:
            action=self.choose_best_action(grid, valid_actions)
        else:
            action = np.random.choice(valid_actions)

        return action
    
    def choose_best_action(self, grid, valid_actions):
        input = np.append(grid, valid_actions)
        input = torch.tensor(input, dtype=torch.float32).to(self.nn.device)
        with torch.no_grad():
            Q_vals = self.nn.forward(input)
            action = torch.argmax(Q_vals).item()
        return action
    
    def get_Q_vals(self, grid, valid_actions):
        input = np.append(grid, valid_actions)
        input = torch.tensor(input, dtype=torch.float32).to(self.nn.device)
        Q_vals = self.evalNN.forward(input)
        return Q_vals
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.nn.optimizer.zero_grad()

        #pick indexes at random
        max_mem = min(self.mem_cntr, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False).astype(int)

        #collect data from indexes
        states = torch.tensor(self.state_memory[batch], dtype=torch.float).to(self.nn.device)
        actions = torch.tensor(self.action_memory[batch]).to(self.nn.device)
        state_actions = torch.stack((states, actions), dim=1)
        

        #attach best move to calculate q_val for new states
        new_valid_moves = torch.tensor(self.new_valid_moves_memory[batch]).to(self.nn.device)
        new_states = torch.tensor(self.new_state_memory[batch]).to(self.nn.device)
        new_states_actions = torch.stack((new_states, new_valid_moves), dim=1)

        #isolate rewards
        rewards = torch.tensor(self.reward_memory[batch]).to(self.nn.device)
        terminals = torch.tensor(self.terminal_memory[batch]).to(self.nn.device)
        
        #train
        self.nn.nn.train()

        #according to current NN, what is the highest q_val on the board? 
        q_eval = torch.max(self.nn.forward(state_actions).squeeze())

        #according to the evalNN, what is the highest q_val for the resulting state? replace terminals with win/lose rewards. 
        q_next                = rewards
        q_next[not terminals] = torch.max(self.evalNN.forward(new_states_actions))
        q_target = rewards + torch.tensor(self.gamma).to(self.nn.device) * q_next

        #update our neural net
        loss = self.nn.loss(q_eval, q_target)
        loss.backward()
        self.nn.optimizer.step()
        self.nn.nn.eval()
        self.counter+=1
        if(self.counter%20==0):
            self.evalNN=copy.deepcopy(self.nn)
        if(self.epsilon > self.eps_end): self.epsilon -= self.eps_dec

    def copy_eval_net(self):
         self.evalNN=copy.deepcopy(self.nn)


        

        





        
