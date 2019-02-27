import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import visdom
from torch.distributions import Categorical

class AI:

    def __init__(self):
        self.policy=Net(2,21)
        self.action_space=np.arange(-1,1.1,0.1)
        self.saved_log_probs=[]
        self.rewards=[]

        self.viz=visdom.Visdom()
        self.plot_v=self.viz.line(X=[0],Y=[0])
        self.plot_a=self.viz.line(X=[0],Y=[0])
        self.v_history=[]
        self.t_history=[]
        self.a_history=[]

    def decision(self, x_gap, v, R, t):  # return A
        self.rewards.append(R)
        probs = self.policy(torch.tensor([[x_gap,v]]))
        dist = Categorical(probs)
        a_index = dist.sample()
        self.saved_log_probs.append(dist.log_prob(a_index))
        a=self.action_space[a_index.item()]

        if(R==1 or R==-1):
            self.finish_episode()

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
    
    def finish_episode(self):
        R=0
        loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std())
        for log_prob, R in zip(self.saved_log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.cat(loss).sum()
        self.policy.optimize(loss)
        del self.rewards[:]
        del self.saved_log_probs[:]

class Net(nn.Module):

    def __init__(self,input_number,output_number):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(input_number, input_number*32)
        self.layer2 = nn.Linear(input_number*32, output_number)
        self.optimizer=optim.Adam(self.parameters(),lr=0.01)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        action_scores = self.layer2(x)
        return F.softmax(action_scores,dim=1)

    def optimize(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
