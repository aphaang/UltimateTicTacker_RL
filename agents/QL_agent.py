import Q_learning_NN as QL
import numpy as np
import torch

def Q_agent():
    def __init__(self, env, input_dims, output_dims, gamma, epsilon, lr, batch_size, n_actions, max_mem_size=10000, 
                 eps_end=0.01, eps_dec=5e-4):
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
        self.nn = QL.Q_learning_NN(input_dims, fc1_dims, fc2_dims, n_actions, output_dims, lr=0.001)
        self.action_space = env.get_action_space()

        self.state_action_memory = np.zeros((self.max_mem_size, input_dims-1), dtype=np.float32)
        self.state_action_memory = np.zeros((self.max_mem_size, input_dims))
        self.new_state_memory = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
        self.new_valid_moves_memory = []*self.max_mem_size

        self.action_memory = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.bool)

    def remember_transition(self, state, action, reward, new_state, new_board, done):
        index = self.mem_cntr % self.max_mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.new_valid_moves_memory[index] = new_board.getListOfPossibleMoves()
        self.state_action_memory[index] = state.join(action)
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def choose_action(self, grid, valid_actions):
        if np.random.random() > self.epsilon:
            self.choose_best_action(self, grid, valid_actions)
        else:
            action = np.random.choice(valid_actions)
        return action
    
    def choose_best_action(self, grid, valid_actions):
        input = [(grid, a) for a in valid_actions]
        input = torch.tensor(input, dtype=torch.float32).to(self.nn.device)
        self.nn.zero_grad()
        actions = self.nn.forward(input)
        actions = torch.argmax(actions).item()
        return actions
    
    def val_of_best_action(self,grid,valid_actions):
        input = [(grid, a) for a in valid_actions]
        input = torch.tensor(input, dtype=torch.float32).to(self.nn.device)
        self.nn.zero_grad()
        actions = self.nn.forward(input)
        return torch.max(actions).item()
    
    def learn(self,env):
        if self.mem_cntr < self.batch_size:
            return
        
        self.nn.optimizer.zero_grad()

        #pick indexes at random
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        #collect data from indexes
        state_actions = torch.tensor(self.state_action_memory[batch]).to(self.nn.device)

        #attach best move to calculate q_val for new states
        new_valid_moves = self.new_valid_moves_memory[batch]
        new_states = torch.tensor(self.new_state_memory[batch]).to(self.nn.device)
        rewards = torch.tensor(self.rewards_memory[batch]).to(self.nn.device)
        terminals = torch.tensor(self.terminal_memory[batch], ).to(self.nn.device)
        actions = self.action_memory[batch]

        batch_index = np.arange(self.batch_size)
        
        self.nn.train()
        q_eval = self.nn.forward(state_actions)
        q_next =[self.val_of_best_action(new_state, valids) for valids, new_state in zip(new_valid_moves,new_states)]
        #q_next[terminals] = 0.0
        
        q_target = rewards + self.gamma * torch.max(q_next)
        
        
        loss = self.nn.loss(q_eval, q_target)
        loss.backward()
        self.nn.optimizer.step()

        if(self.epsilon > self.eps_end): self.epsilon -= self.eps_dec

        





        
