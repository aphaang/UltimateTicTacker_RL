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

    env = TwoPlayerEnv() 
    agentQ = QL_agent(env, 82, 1, 0.1, 1, 0.01, 32, 50)
    agent2 = RandomAgent()
    for i in range(10):
        obs = env.reset()
        done = False
        while not done:
            available = env.valid_actions()
            print(available)
            prev_board, _, _ = obs
            action = agentQ.choose_action(prev_board, available)



            obs, reward, done, _ = env.step(action)
            new_state, _, _ = obs
            agentQ.remember_transition(prev_board, action, reward, new_state, env.pygame.board, done)
            if(agentQ.mem_cntr>64):
                agentQ.learn()

            if done: break
            available = env.valid_actions()
            action = random.choice(available) #agent2.getAction(env, obs, available)
            board, _, _ = obs

            obs, reward, done, _ = env.step(action)
            if done: break
    env.close()