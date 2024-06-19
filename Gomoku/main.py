import torch
from  Agentnetwork import neuralnetwork as NNagent
import utils
import MCTS
from cfg import *

import concurrent.futures
from torch.multiprocessing import Pool, Process, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def main():
    
    Net = NNagent(input_layers=3, board_size=config.board_size, learning_rate=config.learning_rate)
    # Net = torch.load('./model_weight/model_300.pkl')
    buffer = utils.random_stack()
    tree = MCTS.MCTS(board_size=config.board_size,simulation_per_step=300, neural_network=Net)
    Net.adjust_lr(1e-3)
    game_time = 0

    while True:
        # 两个AI 进行对弈
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(tree.game): game_time for _ in range(15)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                game_record, eval, steps = future.result()
            # game_record, eval, steps = tree.game()

                if len(game_record) % 2 == 1:
                    print("game {} completed, black win, this game length is {}".format(game_time, len(game_record)))
                else:
                    print("game {} completed, white win, this game length is {}".format(game_time, len(game_record)))
                print("The average eval:{}, the average steps:{}".format(eval, steps))
                
                # 利用自己对弈的结果生成训练数据
                train_data = utils.generate_training_data(game_record=game_record, board_size=config.board_size)
                for i in range(len(train_data)):
                    buffer.push(train_data[i])
                
                game_time += 1

        my_loader = utils.generate_data_loader(buffer)
        
        # 每50次采样后进行训练，训练5次
        if game_time % 100 == 0:
            for _ in range(5):
                Net.train(my_loader, game_time)
            print("train finished")

        if game_time % 100 == 0 and game_time != 0:
            torch.save(Net, f"model_weight/model_{game_time}.pkl")

        



if __name__ == '__main__':
    main()
    print("Task finish!!!")