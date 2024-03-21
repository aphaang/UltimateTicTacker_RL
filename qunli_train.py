'''
This file is used to generate the data for pre-training the model by 
facing off two random agents against each other. The move, board, and 
valid moves are saved into a file for later use.
'''
import numpy as np

from envs.env_two_player import TwoPlayerEnv
import random 
from agents.QL_agent import QL_agent

class RandomAgent():

    def __init__(self, player = 1):
       self.player = player

    def getAction(self, env, observation, available):
        action = random.choice(available)
        return action

if __name__ == "__main__":
    input_dims=162
    output_dims=81
    gamma=0.1   # possible for abaltion study
    epsilon=0.5   #abalation study
    lr=0.0001     #alation study
    env = TwoPlayerEnv() 
    agentQ = QL_agent(env, input_dims, output_dims, gamma, epsilon, lr, 128, 50)
    agent2 = RandomAgent()
    c=0
    c2=0
    eval_epoch=100
    total_epoch=200
    for i in range(total_epoch):
        obs = env.reset()
        done = False
        print(f"epsilon: {agentQ.epsilon}")
        loss=[]
        while not done:
            available = env.valid_actions()
            #print(available)
            prev_board, _, _ = obs
            action = agentQ.choose_action(prev_board, available)
            


            obs, reward, done, _ = env.step(action)
            new_state, _, _ = obs
            agentQ.remember_transition(prev_board, action, reward, new_state, env.pygame.board, done)
            if(agentQ.mem_cntr>256 and i<eval_epoch):
                loss.append(agentQ.learn().cpu().item())
                
            if done: 
                agentQ.copy_eval_net()
                print(f"the training loss:{np.mean(loss)},reward {reward}")
                if(env.pygame.board.state==1):
                    if(i>=eval_epoch):
                        c+=1
                    print(f"{i} th round,RL agents wins")
                if(env.pygame.board.state==2):
                    if(i>=eval_epoch):
                        c2+=1
                    print(f"{i} th round,random agents wins")
                break
            available = env.valid_actions()
            action = random.choice(available) #agent2.getAction(env, obs, available)
            board, _, _ = obs

            obs, reward, done, _ = env.step(action)
            if done:
                agentQ.copy_eval_net()
                print(f"the training loss:{np.mean(loss)},reward {reward}")
                if(env.pygame.board.state==1):
                    if(i>=eval_epoch):
                        c+=1
                    print(f"{i} th round,RL agents wins")
                if(env.pygame.board.state==2):
                    if(i>=eval_epoch):
                        c2+=1
                    print(f"{i} th round,random agents wins")
                break
    print(f"Rl wins {c} times, random player wins {c2} times, the game draws {total_epoch-eval_epoch-c-c2}")
    env.close()