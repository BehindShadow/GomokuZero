import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Easy_model(nn.Module):
    def __init__(self, input_layer):
        super(Easy_model, self).__init__()
        self.conv1 = nn.Conv2d(input_layer, 16, 3, padding=1) # (input, output, kernel size, padding)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
    def forward(self, input):
        input = self.relu(self.bn1(self.conv1(input)))
        input = self.relu(self.bn2(self.conv2(input)))
        # input = self.relu(self.bn3(self.conv3(input)))

        return input


class Model(nn.Module):
    def __init__(self, input_layer, board_size):
        super(Model, self).__init__()
        self.model = Easy_model(input_layer)
        self.p = 3
        self.output_channel = 32
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # value head
        self.value_conv1 = nn.Conv2d(kernel_size=1, in_channels=self.output_channel, out_channels=8)
        self.value_bn1 = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(in_features=8 * self.p * self.p, out_features=64)
        self.value_fc2 = nn.Linear(in_features=64, out_features=1)
        # policy head
        self.policy_conv1 = nn.Conv2d(kernel_size=1, in_channels=self.output_channel, out_channels=8)
        self.policy_bn1 = nn.BatchNorm2d(8)
        self.policy_fc1 = nn.Linear(in_features=8 * self.p * self.p, out_features=board_size * board_size)

    def forward(self, state):
        s = self.model(state)

        # value head part
        v = self.value_conv1(s)
        v = self.relu(self.value_bn1(v)).view(-1, 8*self.p *self.p)
        v = self.relu(self.value_fc1(v))
        value = self.tanh(self.value_fc2(v))

        # policy head part
        p = self.policy_conv1(s)
        p = self.relu(self.policy_bn1(p)).view(-1, 8*self.p * self.p)
        prob = self.policy_fc1(p)
        return prob, value


class neuralnetwork:
    def __init__(self, input_layers, board_size,  learning_rate=0.1):

        self.model = Model(input_layer=input_layers, board_size=board_size).cuda().double()
        self.opt = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-4)

        self.mse = nn.MSELoss()
        self.crossloss = nn.CrossEntropyLoss()


    def train(self, data_loader, game_time):
        self.model.train()
        loss_record = []
        for batch_idx, (state, distrib, winner) in enumerate(data_loader):
            tmp = []
            state, distrib, winner = Variable(state).double(), Variable(distrib).double(), Variable(winner).unsqueeze(1).double()
            state, distrib, winner = state.cuda(), distrib.cuda(), winner.cuda()

            self.opt.zero_grad()
            prob, value = self.model(state)
            output = F.log_softmax(prob, dim=1)

            cross_entropy = - torch.mean(torch.sum(distrib*output, 1))
            mse = F.mse_loss(value, winner)

            loss = cross_entropy + mse
            loss.backward()

            self.opt.step()
            tmp.append(cross_entropy.data)
            if batch_idx % 10 == 0:
                print(f"We have played {game_time} games, and batch {batch_idx}, the cross entropy loss is {cross_entropy.data}, the mse loss is {mse.data}")
                loss_record.append(sum(tmp)/len(tmp))
        return loss_record


    def eval(self, state):
        self.model.eval()
        state = torch.from_numpy(state).unsqueeze(0).double().cuda()

        with torch.no_grad():
            prob, value = self.model(state)
        return F.softmax(prob, dim=1), value

    def adjust_lr(self, lr):
        for group in self.opt.param_groups:
            group['lr'] = lr