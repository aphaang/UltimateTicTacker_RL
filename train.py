'''
This file is used to generate the data for pre-training the model by 
facing off two random agents against each other. The move, board, and 
valid moves are saved into a file for later use.
'''

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
    input_dims=82
    output_dims=1
    gamma=0.05
    epsilon=0.2
    lr=0.01
    env = TwoPlayerEnv() 
    agentQ = QL_agent(env, input_dims, output_dims, gamma, epsilon, lr, 128, 50)
    agent2 = RandomAgent()
    c=0
    for i in range(100):
        obs = env.reset()
        done = False
        while not done:
            available = env.valid_actions()
            #print(available)
            prev_board, _, _ = obs
            action = agentQ.choose_action(prev_board, available)
            


            obs, reward, done, _ = env.step(action)
            new_state, _, _ = obs
            agentQ.remember_transition(prev_board, action, reward, new_state, env.pygame.board, done)
            if(agentQ.mem_cntr>256 and i<20):
                agentQ.learn()
            if done: 
                print(reward)
                if(reward>0):
                    c+=1
                    print(f"{i} th round,RL agents wins")
                else:
                    print(f"{i} th round,random agents wins")
                break
            available = env.valid_actions()
            action = random.choice(available) #agent2.getAction(env, obs, available)
            board, _, _ = obs

            obs, reward, done, _ = env.step(action)
            if done:
                print(reward)
                if(reward>0):
                    c+=1
                    print(f"{i} th round,RL agents wins")
                else:
                    print(f"{i} th round,random agents wins")
                break
    print(f"Rl wins{c} times")
    env.close()