'''
This file is used to generate the data for pre-training the model by 
facing off two random agents against each other. The move, board, and 
valid moves are saved into a file for later use.
'''

from envs.env_two_player import TwoPlayerEnv
import random 
import numpy as np

class RandomAgent():

    def __init__(self, player = 1):
       self.player = player

    def getAction(self, env, observation, available):
        action = random.choice(available)
        return action
    
def one_hot(valid_actions):
    actions = [0]*81
    for v in valid_actions: 
        actions[v] = 1
    return actions
    

def whole_game(): 
    moves_data = open("moves.txt", "w")
    boards_data = open("boards.txt", "w")
    valid_moves_data = open("valid_moves.txt", "w")

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
            
            
            moves_data.write(str(move_enc) + '\n')
            boards_data.write(str(board) + '\n')
            valid_moves_data.write(str(available) + '\n')

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

'''Function runs through a game and saves a start, mid, and endgame state along with the valid moves '''

def gen_mid_game_state(s, m, e, env):
    obs = env.reset()
    n = 0
    start = None
    mid = None
    end = None
    done = False
    for n in range(e+1):
        available = env.valid_actions()
        action = random.choice(available)
        obs, reward, done, _ = env.step(action)

        available = one_hot(env.valid_actions())
        board, _, _ = obs
        if n == s: start = board, available
        if n == m: mid   = board, available
        if n == e: end   = board, available
        if done: break
    
    return (start, mid, end)

def fan_out(state, n):
    pass

if __name__ == "__main__":
    env = TwoPlayerEnv()
    start_data = open("start_data.txt", "w")
    mid_data = open("mid_data.txt", "w")
    end_data = open("end_data.txt", "w")
    numgames = 4000
    n=0
    while n < numgames:
        n+=1
        s = np.random.randint(0, 12)
        m = np.random.randint(12, 25)
        e = np.random.randint(25, 40)
        (start, mid, end) = gen_mid_game_state(s, m, e, env)
        start_data.write(str(start) + '\n')
        if mid_data == False: 
            n -= 1
            continue
        elif end_data == False:
            mid_data.write(str(mid) + "\n")
            n -= 1
            continue
        
        mid_data.write(str(mid) + '\n')
        end_data.write(str(end) + '\n')
    