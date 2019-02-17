import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import visdom

class AI:

    def __init__(self):
        self.actor = Net(2,1)
        self.critic = Net(2,1)
        self.experience=torch.zeros(0,6)
        self.x_gap=0.
        self.v=0.
        self.a=0.

        self.viz=visdom.Visdom()
        self.plot_v=self.viz.line(X=[0],Y=[0])
        self.plot_a=self.viz.line(X=[0],Y=[0])
        self.v_history=[]
        self.t_history=[]
        self.a_history=[]

    def action(self, x_gap, v):
        return self.actor(torch.tensor([x_gap,v])).item()

    def quality(self, x_gap, v):
        return self.critic(torch.tensor([x_gap,v])).item()

    def decision(self, x_gap, v, R, t, greedy):  # return A
        Q=R+0.9*self.quality(x_gap,v)
        record=torch.tensor([[self.x_gap,self.v,self.a,x_gap,v,Q]])
        temp=torch.pow(self.experience[:,0:3]-record[:,0:3],2)
        distances=torch.sum(temp,1)
        indexes=torch.nonzero(distances<0.0001)
        if len(indexes):
            i=indexes[0].item()
            e=self.experience[i]
            if self.quality(e[3],e[4])<self.quality(x_gap,v):
                self.experience[i]=record
            elif self.experience.size()[0]>600:
                self.experience[random.randint(0,600)]=record
        else:
            if self.experience.size()[0]<=600:
                self.experience=torch.cat((self.experience,record),0)
            else:
                self.experience[random.randint(0,600)]=record

        self.critic.train(self.experience[:,0:2],self.experience[:,5])
        self.actor.train(self.experience[:,0:2],self.experience[:,2])

        a=self.action(x_gap,v)+random.gauss(0,greedy)
        a=max(-1,a)
        a=min(1,a)

        self.x_gap=x_gap
        self.v=v
        self.a=a

        self.v_history.append(v)
        self.a_history.append(a)
        self.t_history.append(t)
        if(R==1 or R==-1):
            self.v_history=[0]
            self.a_history=[0]
            self.t_history=[0]
        if(len(self.v_history)>300):
            del self.v_history[0]
            del self.t_history[0]
            del self.a_history[0]
        self.viz.line(X=self.t_history,Y=self.v_history,win=self.plot_v)
        self.viz.line(X=self.t_history,Y=self.a_history,win=self.plot_a)
        return a

class Net(nn.Module):

    def __init__(self,input_number,output_number):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(input_number, input_number*10)
        self.layer2 = nn.Linear(input_number*10, output_number*10)
        self.layer3 = nn.Linear(output_number*10, output_number)
        self.loss = torch.tensor([1.])
        self.input_number=input_number
        self.output_number=output_number
        self.layer1.weight.data.fill_(0.)
        self.layer2.weight.data.fill_(0.)
        self.layer3.weight.data.fill_(0.)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def train(self,input,target):
        loss_func = nn.MSELoss(reduction='mean')
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        counts = 0
        self.loss = loss_func(self(input), target)
        while not torch.isnan(self.loss) and counts < 20:
            loss = loss_func(self(input), target)
            self.loss = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counts += 1
