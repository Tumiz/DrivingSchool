# 策略梯度求解连续控制问题，小车可以在xy平面上移动，观察量：位姿和车速，控制量：加速度和前轮转角，目标到达目标点并停留

from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh
from torch.nn import Module, Linear
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from agents.functions import normalize, gather, normalsample

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
        w_mu, w_sigma = self.w_head(x1)
        a_mu = tanh(a_mu)
        a_sigma = softplus(a_sigma)
        w_mu = tanh(w_mu)*0.6
        w_sigma = softplus(w_sigma)
        return a_mu, a_sigma, w_mu, w_sigma



class Agent():

    def __init__(self):
        self.policy = Policy()
        self.optimizer = Adam(self.policy.parameters(), lr=0.01)
        self.feeds = []  # records of returns
        self.state_actions = []
        self.a_logprobs = []
        self.w_logprobs = []

        self.viz = visdom.Visdom()
        self.viz.close(win=None)
        self.plot_v = self.viz.line(X=[0], Y=[0])
        self.plot_a = self.viz.line(X=[0], Y=[0])
        self.plot_w = self.viz.line(X=[0], Y=[0])
        self.plot_l = self.viz.line(X=[0], Y=[0])
        self.plot_value = self.viz.line(X=[0], Y=[0])
        self.plot_ac = self.viz.line(X=[0], Y=[0])
        self.plot_wc = self.viz.line(X=[0], Y=[0])
        self.v_history = []
        self.w_history = []
        self.a_history = []
        self.l_history = []

    def judge(self, x, y, v, p_error):
        d = pow(pow(x, 2)+pow(y, 2), 0.5)
        if(d < p_error and v >= 0):
            return 100.
        else:
            return -1.

    def select_action(self,state):
        a_mu,a_sigma,w_mu,w_sigma=self.policy(state)
        a,a_logprob = normalsample(a_mu,a_sigma)
        self.a_logprobs.append(a_logprob)
        w,w_logprob = normalsample(w_mu,w_sigma)
        self.w_logprobs.append(w_logprob)
        self.state_actions.append([state,a,a_mu,a_sigma,w,w_mu,w_sigma])
        return a, w

    def decision(self, done, x, y, rz, v, p_error):  # return action
        feed = self.judge(x, y, v, p_error)
        if len(self.state_actions):
            self.feeds.append(feed)
        if not done:
            state = tensor([x, y, rz, v])
            a, w = self.select_action(state)
        else:
            a = w = 0
            self.finish_episode()

        if(done):
            self.v_history = []
            self.a_history = []
            self.w_history = []
        else:
            self.v_history.append(v)
            self.a_history.append(a)
            self.w_history.append(w)
            x_series = list(range(len(self.v_history)))
            self.viz.line(X=x_series, Y=self.v_history,
                          win=self.plot_v, opts=dict(ylabel="velocity"))
            self.viz.line(X=x_series, Y=self.a_history,
                          win=self.plot_a, opts=dict(ylabel="acceleration"))
            self.viz.line(X=x_series, Y=self.w_history,
                          win=self.plot_w, opts=dict(ylabel="front wheel angle"))
        return a, w

    def finish_episode(self):
        values = gather(self.feeds, 0.99)
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

        nas=[]
        nws=[]
        for state,a,a_mu,a_sigma,w,w_mu,w_sigma in self.state_actions:
            new_a_mu, new_a_sigma, new_w_mu, new_w_sigma = self.policy(state)
            nas.append(tensor([a,a_mu,new_a_mu]))
            nws.append(tensor([w,w_mu,new_w_mu]))
        self.viz.line(X=list(range(len(nas))),Y=stack(nas), win=self.plot_ac, opts=dict(legend=["a","am", "nam"]))
        self.viz.line(X=list(range(len(nws))),Y=stack(nws), win=self.plot_wc, opts=dict(legend=["w","wm", "nwm"]))

        self.l_history.append([loss.item(), sum(self.feeds)])
        self.viz.line(X=list(range(len(self.l_history))), Y=self.l_history,
                      win=self.plot_l, opts=dict(ylabel="loss", legend=["l", "r"]))
        self.viz.line(X=list(range(len(values))), Y=list(zip(nvalues, values, self.feeds)),
                      win=self.plot_value, opts=dict(ylabel="record", legend=["nvalue", "value", "feed"]))

        self.feeds = []
        self.a_logprobs = []
        self.w_logprobs = []
        self.state_actions = []
