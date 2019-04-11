from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh, rand
from torch.nn import Module, Linear
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from random import random
from visdom import Visdom

viz = Visdom()


def normalize(data):  # input tensor
    ret = (data-data.mean())/(data.std()+1e-10)
    return ret


def gather(feeds, gamma=0.99):  # input list
    values = []
    V = 0
    for feed in feeds[::-1]:
        V = feed+gamma*V
        values.insert(0, V)
    return tensor(values)

def calvalues(rewards, gamma=0.99, normalized=False):
    values=gather(rewards,gamma)
    if normalized:
        values=normalize(values)
    return values

def totalvalue(rewards, gamma=0.99):
    return calvalues(rewards, gamma).sum()

def processlogprob(logprobs):
    return sum(logprobs)

def truncatedsample(samplefunc, low, high):
    sample = samplefunc()
    while sample <= low or sample >= high:
        sample = samplefunc()
    return sample


def normalsample(mu, sigma):
    dist = Normal(mu, sigma)
    sample = dist.sample().item()
    logprob = dist.log_prob(sample)
    return sample, logprob


class Model(Module):
    def __init__(self, nin, nout):
        super(Model, self).__init__()
        self.layer1 = Linear(nin, nin*nout*4)
        self.layer2 = Linear(nin*nout*4, nout)
        self.optimizer = Adam(self.parameters())
        self.loss = tensor(0., requires_grad=True)

    def forward(self, x):
        x = elu(self.layer1(x))
        return self.layer2(x)

    def optimize(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class Car:
    def __init__(self):
        self.a = 0.
        self.t = 0.
        self.v = 0.
        self.x = 0.
        # self.plot=viz.line(X=[0],Y=[[self.a,self.v,self.x]])

    def step(self, t):
        dt=t-self.t
        self.v += self.a*dt
        self.x += self.v*dt
        # viz.line(X=[t],Y=[[self.a,self.v,self.x]],win=self.plot,update='append')

    def reset(self):
        self.a=0.
        self.v=0.
        self.x=0.
        self.t=0.


class Environment:
    def __init__(self):
        self.t = 0.
        self.car = Car()
        self.state = tensor([self.car.v,self.car.x])
        self.state_distance=0
        self.done = False

    def step(self, a):
        self.car.a = a
        self.car.step(self.t)
        self.t += 0.01
        self.state = tensor([self.car.v,self.car.x])
        self.state_distance=self.state.norm()
        if(self.t>0.5):
            self.done=True
            return self.state_distance
        else:
            return 0

    def reset(self):
        self.t=0.
        self.car.reset()
        self.car.x=random()*10
        self.state=tensor([self.car.v,self.car.x])
        self.state_distance=self.state.norm()
        self.done=False


def tonormal(a1, a2):
    return tanh(a1), softplus(a2)

class Simulation:

    def __init__(self):
        self.env = Environment()
        self.m = Model(2, 2)

    def run(self,times,plotinterval=1000):
        for i in range(times):
            rewards=[]
            a_probs=[]
            series = []
            self.env.reset()
            while not self.env.done:
                d=self.m(self.env.state)
                a_mu, a_sigma = tonormal(d[0], d[1])
                a, a_prob = normalsample(a_mu, a_sigma)
                reward = self.env.step(a)
                rewards.append(reward)
                a_probs.append(a_prob)
                series.append([a_mu.item(), a_sigma.item(), self.env.car.v, self.env.car.x, self.env.state_distance])

            self.m.loss=totalvalue(rewards)*processlogprob(a_probs)
            self.m.optimize()
            if(i%plotinterval==0):
                viz.line(X=list(range(len(series))), Y=series, opts=dict(
                    legend=["a_mu", "a_sigma", "v", "x", "d"]))
