from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh, rand
from torch.nn import Module, Linear
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from random import random
from math import cos, sin, tan, pi, floor
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

def wraptopi(x):
    x = x - floor(x/(2*pi)) *2 *pi
    if(x>pi and x<2*pi):
        x=x-2*pi
    if x<-pi and x>-2*pi:
        x=2*pi+x
    return x

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
        self.w = 0.
        self.t = 0.
        self.v = 0.
        self.x = 0.
        self.y = 0.
        self.h = 0.
        # self.plot=viz.line(X=[0],Y=[[self.a,self.v,self.x]])

    def step(self, t):
        dt=t-self.t
        vx = self.v*cos(self.h)
        vy = self.v*sin(self.h)
        self.x += vx*dt
        self.y += vy*dt
        self.v += self.a*dt
        self.h += self.v/2.7*tan(self.w)*dt
        self.h = wraptopi(self.h)
        # viz.line(X=[t],Y=[[self.a,self.v,self.x]],win=self.plot,update='append')

    def reset(self):
        self.a=0.
        self.w=0.
        self.v=0.
        self.x=0.
        self.y=0.
        self.h=0.
        self.t=0.

    def state(self):
        return tensor([self.v,self.x,self.y,self.h])


class Environment:
    def __init__(self):
        self.t = 0.
        self.car = Car()
        self.car.x=-5
        self.car.y=(random()-1)*5
        self.car.h=random()*pi*0.5
        self.state=self.car.state()
        self.state_distance=self.state.norm()
        self.done = False

    def step(self, a, w):
        self.car.a = a
        self.car.w = w
        self.car.step(self.t)
        self.t += 0.01
        self.state = self.car.state()
        self.state_distance=self.state.norm()
        if(self.t>0.5 or self.state_distance>30):
            self.done=True
            return self.state_distance
        else:
            return 0

class Simulation:

    def __init__(self):
        self.env = Environment()
        self.m = Model(4, 4)

    def run(self,times,plotinterval=1000):
        for i in range(times):
            rewards=[]
            probs=[]
            actions = []
            states=[]
            self.env.__init__()
            while not self.env.done:
                d=self.m(self.env.state)
                a_mu, a_sigma = tanh(d[0]),softplus(d[1])
                a, a_prob = normalsample(a_mu, a_sigma)
                w_mu, w_sigma = 0.5*tanh(d[2]),softplus(d[3])
                w, w_prob = normalsample(w_mu, w_sigma)
                reward = self.env.step(a,w)
                rewards.append(reward)
                probs.append(a_prob+w_prob)
                actions.append([a_mu.item(), a_sigma.item(), w_mu.item(),w_sigma.item()])
                states.append([self.env.car.v, self.env.car.x, self.env.car.y, self.env.car.h, self.env.state_distance])

            self.m.loss=totalvalue(rewards)*processlogprob(probs)
            self.m.optimize()
            if(i%plotinterval==0):
                viz.line(X=list(range(len(actions))), Y=actions, opts=dict(
                    legend=["a_mu", "a_sigma","w_mu","w_sigma"]))
                    
                viz.line(X=list(range(len(states))), Y=states, opts=dict(legend=["v", "x", "y", "h", "d"]))

def testcar():
    car=Car()
    car.x=-5
    car.y=-5
    car.v=1
    car.w=0.5
    xs=[]
    ys=[]
    for t in range(100):
        car.step(t*0.01)
        xs.append(car.x)
        ys.append(car.y)
    viz.quiver(X=xs,Y=ys)

viz.close(win=None)
sim=Simulation()
sim.run(10001,1000)