from torch.optim import Adam
from torch import tensor, arange, stack, isnan
from torch.nn import Module, Linear
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from algris import normalize, gather

import visdom


class Policy(Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.s_head = Linear(4, 4*16)  # accept state
        # a normal distribution represents output of 'a', mu,sigma
        self.a_head = Linear(4*16, 2)
        # a normal distribution represents output of 'w', mu,sigma
        self.w_head = Linear(4*16, 2)

    def forward(self, x):
        x1 = elu(self.s_head(x))
        a_mu, a_sigma = self.a_head(x1)
        a_sigma = softplus(a_sigma)
        a_dist = Normal(a_mu, a_sigma)
        a = a_dist.sample()
        a_logprob = a_dist.log_prob(a)
        w_mu, w_sigma = self.w_head(x1)
        w_sigma = softplus(w_sigma)
        w_dist = Normal(w_mu, w_sigma)
        w = w_dist.sample()
        w_logprob = w_dist.log_prob(w)
        if isnan(a) or isnan(w):
            print(x, a_mu, a_sigma, a, a_logprob, w_mu, w_sigma, w, w_logprob)
        return a.item(), a_logprob, w.item(), w_logprob


class Agent(Module):

    def __init__(self):
        super(Agent, self).__init__()
        self.policy = Policy()
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.feeds = []  # records of returns
        self.states = []
        self.a_logprobs = []
        self.w_logprobs = []

        self.viz = visdom.Visdom()
        self.viz.close(win=None)
        self.plot_v = self.viz.line(X=[0], Y=[0])
        self.plot_a = self.viz.line(X=[0], Y=[0])
        self.plot_w = self.viz.line(X=[0], Y=[0])
        self.plot_r = self.viz.line(X=[0], Y=[0])
        self.plot_l = self.viz.line(X=[0], Y=[0])
        self.plot_value = self.viz.line(X=[0], Y=[0])
        self.v_history = []
        self.w_history = []
        self.a_history = []
        self.f_history = []
        self.l_history = []

    def judge(self, x, y, v, p_error):
        d = pow(pow(x, 2)+pow(y, 2), 0.5)
        if(d < p_error and v >= 0):
            return 100
        elif v<0:
            return v
        else:
            return -1

    def decision(self, done, x, y, rz, v, p_error):  # return action
        feed = self.judge(x, y, v, p_error)
        if len(self.states):
            self.feeds.append(feed)
        if not done:
            state = tensor([x, y, rz, v])
            a, a_logprob, w, w_logprob = self.policy(state)
            self.states.append(state)
            self.a_logprobs.append(a_logprob)
            self.w_logprobs.append(w_logprob)
        else:
            a = w = 0
            self.finish_episode()

        if(done):
            self.v_history = []
            self.a_history = []
            self.w_history = []
            self.f_history = []
        else:
            self.v_history.append(v)
            self.a_history.append(a)
            self.w_history.append(w)
            self.f_history.append(feed)
            x_series = list(range(len(self.v_history)))
            self.viz.line(X=x_series, Y=self.v_history,
                          win=self.plot_v, opts=dict(ylabel="velocity"))
            self.viz.line(X=x_series, Y=self.a_history,
                          win=self.plot_a, opts=dict(ylabel="acceleration"))
            self.viz.line(X=x_series, Y=self.w_history,
                          win=self.plot_w, opts=dict(ylabel="front wheel angle"))
            self.viz.line(X=x_series, Y=self.f_history,
                          win=self.plot_r, opts=dict(ylabel="feed"))
        return a, w

    def finish_episode(self):
        values, rloss = gather(self.feeds, 0.99)
        tvalues = tensor(values)
        nvalues = normalize(tvalues)
        losses = []
        for nvalue, a_logprob, w_logprob in zip(nvalues, self.a_logprobs, self.w_logprobs):
            # log(1/p(a,w))=-log(p(a)*p(w))=-log(p(a))-log(p(w))
            losses.append(nvalue*(-a_logprob-w_logprob))
        loss = stack(losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.l_history.append(
            [loss.item(), rloss])
        self.viz.line(X=list(range(len(self.l_history))), Y=self.l_history,
                      win=self.plot_l, opts=dict(ylabel="loss", legend=["l", "r"]))
        self.viz.line(X=list(range(len(values))), Y=list(zip(nvalues, values, self.feeds)),
                      win=self.plot_value, opts=dict(ylabel="record", legend=["nvalue", "value", "feed"]))

        self.feeds = []
        self.a_logprobs = []
        self.w_logprobs = []
        self.states = []
