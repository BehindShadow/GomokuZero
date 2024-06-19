import torch

class config():
    temperature = 1
    Cpuct = 0.1
    batch_size = 32
    board_size = 9
    learning_rate = 0.001
    buffer_size = 1024

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'