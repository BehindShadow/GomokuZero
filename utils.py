import numpy as np
import torch
import torch.utils.data as torch_data
from cfg import *


num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

char2num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
            "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20}



class distribution_calculater:
    def __init__(self, size):
        self.map = {} # 命名之后全部值设置为0， 这两个加起来不就是有序字典么
        self.order = [] # aa, ab, ac, .... cc. for 3*3 board size
        for i in range(size):
            for j in range(size):
                name = num2char[i]+num2char[j]
                self.order.append(name)
                self.map[name] = 0

    def push(self, key, value):
        self.map[key] = value

    def get(self, train=True):
        result = []
        choice_pool = []
        choice_prob = []
        for key in self.order:
            if self.map[key] != 0:
                choice_pool.append(key)
                tmp = np.float_power(self.map[key], 1 / config.temperature)
                choice_prob.append(tmp)
                result.append(tmp)
                self.map[key] = 0
            else:
                result.append(0)

        he = sum(result)
        for i in range(len(result)):
            if result[i]:
                result[i] = result[i] / he
        choice_prob = [choice/he for choice in choice_prob]
        if train:
            move = np.random.choice(choice_pool, p=0.8 * np.array(choice_prob) + 0.2 * np.random.dirichlet(0.3*np.ones(len(choice_prob))))
        else:
            move = choice_pool[np.argmax(choice_prob)]
        return move, result


def move_to_str(action):
    return num2char[action[0]] + num2char[action[1]]

def str_to_move(str):
    return np.array([char2num[str[0]], char2num[str[1]]])

def valid_move(state):
    return list(np.argwhere(state==0))


class random_stack:
    def __init__(self, length=1024):
        self.state = []
        self.distrib = []
        self.winner = []
        self.length = length

    def isEmpty(self):
        return len(self.state) == 0

    def push(self, item):
        self.state.append(item["state"])
        self.distrib.append(item["distribution"])
        self.winner.append(item["value"])
        if len(self.state)>= self.length:
            self.state = self.state[1:]
            self.distrib = self.distrib[1:]
            self.winner = self.winner[1:]

    def seq(self):
        return self.state, self.distrib, self.winner


def generate_training_data(game_record, board_size):
    board = np.zeros([board_size, board_size])
    data = []
    player = 1
    winner = -1 if len(game_record) % 2 == 0 else 1
    for i in range(len(game_record)):
        step = str_to_move(game_record[i]['action'])
        state = transfer_to_input(board, player, board_size)
        data.append({"state": state, "distribution": game_record[i]['distribution'], "value": winner})
        board[step[0], step[1]] = player
        player, winner = -player, -winner
    return data


def generate_data_loader(stack):
    state, distrib, winner = stack.seq()
    tensor_x = torch.stack(tuple([torch.from_numpy(s) for s in state]))
    tensor_y1 = torch.stack(tuple([torch.Tensor(y1) for y1 in distrib]))
    tensor_y2 = torch.stack(tuple([torch.Tensor([float(y2)]) for y2 in winner]))
    dataset = torch_data.TensorDataset(tensor_x, tensor_y1, tensor_y2)
    my_loader = torch_data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return my_loader

def transfer_to_input(state, current_player, board_size):
    if current_player==1:
        tmp3 = np.ones([board_size, board_size]).astype(float)
        tmp2 = np.array(state > 0).astype(float)
        tmp1 = np.array(state < 0).astype(float)
    else:
        tmp3 = np.zeros([board_size, board_size])
        tmp2 = np.array(state < 0).astype(float)
        tmp1 = np.array(state > 0).astype(float)
    return np.stack([tmp1, tmp2, tmp3])


