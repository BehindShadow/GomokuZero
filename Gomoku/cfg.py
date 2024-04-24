import torch

class config():
    temperature = 1
    Cpuct = 0.1
    batch_size = 64
    board_size = 15
    learning_rate = 0.02
    buffer_size = 1024

    device = 'cuda' if torch.cuda.is_available() else 'cpu'