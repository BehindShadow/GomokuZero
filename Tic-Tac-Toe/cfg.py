import torch

class config():
    temperature = 1
    Cpuct = 0.1
    batch_size = 256
    board_size = 9
    learning_rate = 0.01
    buffer_size = 1536

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'