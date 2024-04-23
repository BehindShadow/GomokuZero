import torch

class config():
    temperature = 1
    Cpuct = 0.1
    batch_size = 64
    board_size = 3
    learning_rate = 0.02

    device = 'cuda' if torch.cuda.is_available() else 'cpu'