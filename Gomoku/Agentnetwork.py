import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from cfg import config


class BasicBlock(nn.Module):

    def __init__(self, input_channel, output_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride,padding = 1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1,padding = 1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, input_layer):# Block 类型， block数量， 输入通道数
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(input_layer, 32, kernel_size=3, stride=1, padding=1,
                               bias=False) # 3*15*15 -> 16*15*15
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0]) # 16*15*15 -> 32*15*15 
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x



class Model(nn.Module):
    def __init__(self, input_layer, board_size):
        super(Model, self).__init__()
        self.model =  ResNet(block=BasicBlock, layers=[2, 2, 2, 2], input_layer=input_layer)
        self.p = 4
        self.output_channel = 128
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # value head
        self.value_conv1 = nn.Conv2d(kernel_size=1, in_channels=self.output_channel, out_channels=32)
        self.value_bn1 = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(in_features=32 * self.p * self.p, out_features=256)
        self.value_fc2 = nn.Linear(in_features=256, out_features=1)
        # policy head
        self.policy_conv1 = nn.Conv2d(kernel_size=1, in_channels=self.output_channel, out_channels=32)
        self.policy_bn1 = nn.BatchNorm2d(32)
        self.policy_fc1 = nn.Linear(in_features=32 * self.p * self.p, out_features=board_size * board_size)

    def forward(self, state):
        s = self.model(state)
        # value head part
        v = self.value_conv1(s)
        v = self.relu(self.value_bn1(v)).view(-1, 32*self.p *self.p)
        v = self.relu(self.value_fc1(v))
        value = self.tanh(self.value_fc2(v))

        # policy head part
        p = self.policy_conv1(s)
        p = self.relu(self.policy_bn1(p)).view(-1, 32*self.p * self.p)
        prob = self.policy_fc1(p)
        return prob, value


class neuralnetwork:
    def __init__(self, input_layers, board_size,  learning_rate=0.1):

        self.model = Model(input_layer=input_layers, board_size=board_size).to(config.device).double()
        self.opt = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-4)

        self.mse = nn.MSELoss()
        self.crossloss = nn.CrossEntropyLoss()


    def train(self, data_loader, game_time):
        self.model.train()
        loss_record = []
        for batch_idx, (state, distrib, winner) in enumerate(data_loader):
            tmp = []
            state, distrib, winner = Variable(state).double(), Variable(distrib).double(), Variable(winner).unsqueeze(1).double()
            state, distrib, winner = state.to(config.device), distrib.to(config.device), winner.to(config.device)

            self.opt.zero_grad()
            prob, value = self.model(state)
            output = F.log_softmax(prob, dim=1)

            cross_entropy = - torch.mean(torch.sum(distrib*output, 1))
            mse = F.mse_loss(value, winner.squeeze(1))

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
        state = torch.from_numpy(state).unsqueeze(0).double().to(config.device)

        with torch.no_grad():
            prob, value = self.model(state)
        return F.softmax(prob, dim=1), value

    def adjust_lr(self, lr):
        for group in self.opt.param_groups:
            group['lr'] = lr