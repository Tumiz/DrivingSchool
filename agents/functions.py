from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh
from torch.nn import Module, Linear
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from visdom import Visdom

viz=Visdom()

def normalize(data):# input tensor
    ret = (data-data.mean())/(data.std()+1e-10)
    return ret

def gather(feeds,gamma):#input list
    values=[]
    V=0
    for feed in feeds[::-1]:
        V=feed+gamma*V
        values.insert(0,V)
    return values

def truncatedsample(samplefunc,low,high):
    sample=samplefunc()
    while sample<=low or sample>=high:
        sample=samplefunc()
    return sample

def plot(x,y):
    viz.line(X=x,Y=y)

def normalsample(mu,sigma):
    dist = Normal(mu, sigma)
    sample = dist.sample().item()
    logprob = dist.log_prob(sample)
    return sample, logprob

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.s_head = Linear(4, 4*16)  # accept state
        # a normal distribution represents output of 'a', mu,sigma
        self.a_head = Linear(4*16, 2)
        self.optimizer = Adam(self.parameters())
        self.loss = tensor(0.,requires_grad=True)

    def forward(self, x):
        x1 = elu(self.s_head(x))
        a_mu, a_sigma = self.a_head(x1)
        a_mu = tanh(a_mu)
        a_sigma = softplus(a_sigma)
        return a_mu, a_sigma

    def optimize(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

def test(episode):
    m=Model()
    series=[]
    state=tensor([0.,0.,0.,0.])
    for i in range(episode):
        d=m(state)
        series.append(d[0].item())
        s=normalsample(d[0],d[1])
        m.loss=-s[0]*-s[1]
        m.optimize()
    plot(list(range(len(series))),series)