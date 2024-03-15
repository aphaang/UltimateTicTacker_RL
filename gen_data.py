'''
This file is used to generate the data for pre-training the model by 
facing off two random agents against each other. The move, board, and 
valid moves are saved into a file for later use.
'''

from envs.env_two_player import TwoPlayerEnv
import random 

class RandomAgent():

    def __init__(self, player = 1):
       self.player = player

    def getAction(self, env, observation, available):
        action = random.choice(available)
        return action

if __name__ == "__main__":

    moves_data = open("moves.txt", "w")
    boards_data = open("boards.txt", "w")
    valid_moves_data = open("valid_moves.txt", "w")

    env = TwoPlayerEnv()
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    for i in range(100):
        obs = env.reset()
        done = False
        while not done:
            
            available = env.valid_actions()
            print(available)
            action = random.choice(available) #agent1.getAction(env, obs, available)
            move_enc = [0]*81
            move_enc[action] = 1
            board, _, _ = obs
            
            
            moves_data.write(str(move_enc) + "\n")
            boards_data.write(str(board) + "\n")
            valid_moves_data.write(str(available) + "\n")

            obs, reward, done, _ = env.step(action)

            if done: break
            available = env.valid_actions()
            action = random.choice(available) #agent2.getAction(env, obs, available)
            board, _, _ = obs
            
            #save to corresponding files
            move_enc = [0]*81
            move_enc[action] = 1
            moves_data.write(str(move_enc) + "\n")
            boards_data.write(str(board) + "\n")
            valid_moves_data.write(str(available) + "\n")

            obs, reward, done, _ = env.step(action)
            if done: break
    env.close()