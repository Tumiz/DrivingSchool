import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import visdom

class AI:

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.x_gap=0
        self.v=0
        self.a=0

        self.viz=visdom.Visdom()
        self.plot_velocity=self.viz.line(X=[0],Y=[0])
        self.v_history=[]
        self.t_history=[]

    def decision(self, x_gap, v, R, t, greedy):  # return A
        Q=self.critic.value()
        self.actor.update()

        self.v_history.append(v)
        self.t_history.append(t)
        if(len(self.v_history)>300):
            del self.v_history[0]
            del self.t_history[0]
        self.viz.line(X=self.t_history,Y=self.v_history,win=self.plot_velocity)
        return self.v_target

class Actor(Net):
    def __init__(self):
        super(Actor, self).__init__(2, 1, 1000)
        self.x_gap=0
        self.v=0
        self.a=0
        
    def value(self,x_gap,v):
        return self(torch.tensor([x_gap,v])).item()

    def update(self,Q):
        for i,e in enumerate(self.experience):
            if pow(e[0]-self.x_gap)+pow(e[1]-self.v)+pow(e[2]-self.a)<0.0001 and e[3]<Q:
                del self.experience[i]
        self.experience.append([self.x_gap, self.v, self.a, Q])
        self.train()

    def decision(self,x_gap,v,greedy):
        self.a=self.value(x_gap,v)+random.gauss(0,greedy)
        self.a=min(1,self.a)
        self.a=max(-1,self.a)
        self.x_gap=x_gap
        self.v=v

class Critic(Net):
    def __init__(self):
        super(Critic, self).__init__(3, 1, 1000)

    def value(self,x_gap,v,a):
        return self(torch.tensor([x_gap,v,a])).item()

    def update(self,R):


class Net(nn.Module):

    def __init__(self,input_number,output_number,experience_pool_size):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(input_number, input_number*10)
        self.layer2 = nn.Linear(input_number*10, output_number*10)
        self.layer3 = nn.Linear(output_number*10, output_number)
        self.experience = []
        self.loss = torch.tensor([1.])
        self.input_number=input_number
        self.output_number=output_number
        self.experience_pool_size=experience_pool_size

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def train(self):
        l = len(self.experience)
        if(l == 0):
            return
        if(l > self.experience_pool_size):
            self.experience=random.sample(self.experience, self.experience_pool_size)
        sample = torch.tensor(self.experience)
        loss_func = nn.MSELoss(reduction='mean')
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        input = sample[:, 0:self.input_number].float()
        target = sample[:, self.input_number:self.input_number+self.output_number].float()
        counts = 0
        self.loss = loss_func(self(input), target)
        while not torch.isnan(self.loss) and counts < 10:
            loss = loss_func(self(input), target)
            self.loss = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counts += 1
