import torch
import numpy as np
from  Agentnetwork import neuralnetwork as NNagent
import utils
from cfg import *

def main(board):
    
    Net = NNagent(input_layers=3, board_size=config.board_size, learning_rate=config.learning_rate)
    Net = torch.load('./model_weight/model_650.pkl')

    state = board
    state_prob, _ = Net.eval(utils.transfer_to_input(state, 1, 3))
    # print(state_prob)
    next_step = (-1,-1)
    mx = -1
    for r in range(3):
        for l in range(3):
            print(f"row: {r + 1}, col: {l + 1}, {state_prob[0,r*3 + l]}")
            if state_prob[0,r*3 + l] > mx and board[r][l] == 0:
                mx = state_prob[0,r*3 + l]
                next_step = (r + 1, l + 1)

    print(f"下一步的位置应该在：({next_step[0]}, {next_step[1]}) 处")

if __name__ == '__main__':
    main(np.array([[0,0,1],[0,1,0],[0,-1,-1]]))