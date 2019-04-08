from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh
from torch.nn import Module, Linear
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from visdom import Visdom

viz = Visdom()


def normalize(data):  # input tensor
    ret = (data-data.mean())/(data.std()+1e-10)
    return ret


def gather(feeds, gamma):  # input list
    values = []
    V = 0
    for feed in feeds[::-1]:
        V = feed+gamma*V
        values.insert(0, V)
    return values


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
        self.a = 0
        self.v = 0
        self.w = 0
        self.x = 0

    def step(self, dt):
        self.v += self.a*dt
        self.x += self.v*dt


class Environment:
    def __init__(self):
        self.car = Car()

    def step(self, a, w):
        self.a = a
        self.w = w
        self.car.step(0.01)
        return abs(self.car.x-5)


def tonormal(a1, a2):
    return tanh(a1), softplus(a2)


env = Environment()
m = Model(4, 4)
series = []
state = tensor([0., 0., 0., 0.])
for i in range(4000):
    d = m(state)
    a_mu, a_sigma = tonormal(d[0], d[1])
    w_mu, w_sigma = tonormal(d[2], d[3])
    a = normalsample(a_mu, a_sigma)
    w = normalsample(w_mu, w_sigma)
    m.loss = env.step(a[0], w[0])*(a[1]+w[1])
    m.optimize()
    series.append([a_mu.item(), a_sigma.item(), w_mu.item(), w_sigma.item()])
viz.line(X=list(range(len(series))), Y=series, opts=dict(
    legend=["a_mu", "a_sigma", "w_mu", "w_sigma"]))
