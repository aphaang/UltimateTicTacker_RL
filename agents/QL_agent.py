from agents.NN import NN
import numpy as np
import torch
import copy

class QL_agent():
    def __init__(self, env, input_dims, output_dims, gamma, epsilon, lr, batch_size, n_actions, max_mem_size=1000, 
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
        self.nn = NN(81*2, fc1_dims=256, fc2_dims=256, output_dims=output_dims, lr=self.lr)
        self.evalNN=copy.deepcopy(self.nn)
        self.state_memory = np.zeros((self.max_mem_size, input_dims-1), dtype=np.float32)
        self.state_action_memory = np.zeros((self.max_mem_size, 81*2))
        self.new_state_memory = np.zeros((self.max_mem_size, input_dims-1), dtype=np.float32)
        self.new_valid_moves_memory = [0]*self.max_mem_size

        self.action_memory = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.bool)

    def remember_transition(self, state, action, reward, new_state, new_board, done):
        index = self.mem_cntr % self.max_mem_size
        self.state_memory[index] = copy.deepcopy(state)
        self.new_state_memory[index] = copy.deepcopy(new_state)
        self.new_valid_moves_memory[index] = copy.deepcopy(new_board.getListOfPossibleMoves())
        old_state=copy.deepcopy(state)
        one_hot= np.zeros(81)
        one_hot[action]=1
        self.state_action_memory[index] = np.append(old_state,one_hot)
        self.reward_memory[index] = copy.deepcopy(reward)
        self.action_memory[index] = copy.deepcopy(action)
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def choose_action(self, grid, valid_actions):
        if np.random.random() > self.epsilon:
            action=self.choose_best_action(grid, valid_actions)
        else:
            action = np.random.choice(valid_actions)
        
        return action
    
    def choose_best_action(self, grid, valid_actions):
        #print(type(grid))
        #print(type(valid_actions))
        #for a in valid_actions:
         #   print(grid.append(a))
        input=[]
        for a in valid_actions:
            one_hot= np.zeros(81)
            one_hot[a]=1
            input.append(np.append(grid,one_hot))
        #input = [np.append(grid, np.zeros(81)) for a in valid_actions]
        #print(input)
        input = torch.tensor(input, dtype=torch.float32).to(self.nn.device)
        with torch.no_grad():
            actions = self.nn.forward(input).squeeze()
            #print(actions.shape)
            actions = torch.argmax(actions).item()
        return valid_actions[actions]
    
    def val_of_best_action(self,grid,valid_actions):
        #print(grid)
        #print(valid_actions)
        input=[]
        for a in valid_actions:
            one_hot= np.zeros(81)
            one_hot[a]=1
            input.append(np.append(grid,one_hot))        
        #input = [np.append(grid, a) for a in valid_actions]
        input = torch.tensor(input, dtype=torch.float32).to(self.nn.device)
        actions = self.evalNN.forward(input)
        return torch.max(actions).item()
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.nn.optimizer.zero_grad()

        #pick indexes at random
        max_mem = min(self.mem_cntr, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False).astype(int)

        #collect data from indexes
        state_actions = torch.tensor(self.state_action_memory[batch], dtype=torch.float).to(self.nn.device)

        #attach best move to calculate q_val for new states
        new_valid_moves = [self.new_valid_moves_memory[n] for n in batch]
        new_states = self.new_state_memory[batch]
        rewards = torch.tensor(self.reward_memory[batch]).to(self.nn.device)
        terminals = torch.tensor(self.terminal_memory[batch]).to(self.nn.device)
        actions = self.action_memory[batch]
     
        batch_index = np.arange(self.batch_size)
        
        self.nn.nn.train()
        q_eval = self.nn.forward(state_actions).squeeze()
        q_next=[]
        for i in range(len(batch)):
            if(terminals[i]):
                q_next.append(rewards[i])
            else:
                q_next.append(self.val_of_best_action(new_states[i],new_valid_moves[i]))
            
        
        
        q_target = rewards + torch.tensor(self.gamma).to(self.nn.device) * torch.tensor(q_next).to(self.nn.device)
        #print(q_target.shape)
        #print(q_eval.shape)
        
        loss = self.nn.loss(q_eval, q_target)
        loss.backward()
        self.nn.optimizer.step()
        self.nn.nn.eval()
        self.counter+=1
        if(self.epsilon > self.eps_end): self.epsilon -= self.eps_dec
        return loss
    
    def copy_eval_net(self):
        self.evalNN=copy.deepcopy(self.nn)

    def save_model(self, id=0):
        self.evalNN.save("QL_agent_" + id)
        
    def load_model(self, id=0):
        self.evalNN.load("QL_agent" + id)

        

        

        





        
