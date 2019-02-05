import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AI:
    def __init__(self):
        self.planner = Planner()
        self.controller = Controller()
        self.destination = 20

    def decision(self, x_gap, v):
        v_target = self.planner.decision(x_gap)
        a = self.controller.decision(v-v_target)
        return a


class Planner(nn.Module):

    def __init__(self):
        super(Planner, self).__init__()
        self.layer1 = nn.Linear(1, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 1)
        self.x_gap = 0.
        self.v_target = 0.  # target velocity
        self.experience = []
        self.loss = torch.tensor([1.])

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def q(self, x_gap):  # return Q
        return self(torch.tensor([x_gap])).item()

    def decision(self, x_gap):  # return A
        if(abs(x_gap) < abs(self.x_gap)):
            self.experience.append([self.x_gap, self.v_target])
            self.v_target = self.q(x_gap)
        else:
            self.v_target = random.uniform(-1, 10)
        self.x_gap = x_gap
        self.train()
        return self.v_target

    def train(self):
        l = len(self.experience)
        if(l == 0):
            return
        if(l > 1000):
            self.experience=random.sample(self.experience, 1000)
        sample = torch.tensor(self.experience)
        loss_func = nn.MSELoss(reduction='mean')
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        input = sample[:, 0:1].float()
        target = sample[:, 1:2].float()
        counts = 0
        self.loss = loss_func(self(input), target)
        while not torch.isnan(self.loss) and counts < 10:
            loss = loss_func(self(input), target)
            self.loss = loss
            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            optimizer.step()
            counts += 1


class Controller(nn.Module):

    def __init__(self):
        super(Controller, self).__init__()
        self.layer1 = nn.Linear(1, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 1)
        self.v_gap = 0.  # gap between real velocity and target velocity
        self.a = 0.  # acc
        self.experience = []
        self.loss = torch.tensor([1.])

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def reached(self, v_gap):
        return v_gap < 0.01 and self.a < 0.01

    def q(self, v):  # return Q
        return self(torch.tensor([v])).item()

    def decision(self, v_gap):  # input gap to target velocity, output acc
        if abs(v_gap) < abs(self.v_gap):
            self.experience.append([self.v_gap, self.a])      
            self.a = self.q(v_gap)
        else:
            self.a = random.uniform(-1, 1)
        self.v_gap = v_gap
        self.train()
        print(v_gap,self.a,)
        return self.a

    def train(self):
        l = len(self.experience)
        if(l == 0):
            return
        if(l > 1000):
            self.experience=random.sample(self.experience, 1000)
        sample = torch.tensor(self.experience)
        loss_func = nn.MSELoss(reduction='mean')
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        input = sample[:, 0:1].float()
        target = sample[:, 1:2].float()
        counts = 0
        self.loss = loss_func(self(input), target)
        while not torch.isnan(self.loss) and counts < 10:
            loss = loss_func(self(input), target)
            self.loss = loss
            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            optimizer.step()
            counts += 1
