from torch.optim import Adam
from torch import tensor, arange
from torch.nn import Module, Linear
from torch.nn.functional import softmax, elu
from torch.distributions import Categorical
from algris import standardize

import visdom


class Agent(Module):

    def __init__(self):
        super(Agent, self).__init__()
        self.a_space = arange(-1, 1.1, 0.2)
        self.w_space = arange(-0.5, 0.6, 0.1)
        self.s_head = Linear(4, 4*16)
        self.a_head = Linear(4*16, len(self.a_space))
        self.w_head = Linear(4*16, len(self.w_space))
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.records = []  # records of returns

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
        self.r_history = []
        self.l_history = []

    def forward(self, x):
        x = elu(self.s_head(x))
        a_scores = self.a_head(x)
        w_scores = self.w_head(x)
        return softmax(a_scores, dim=0), softmax(w_scores, dim=0)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decision(self, done, x, y, rz, v, r, t):  # return action
        if len(self.records):
            self.records[-1][0]=r
        if not done:
            a_probs, w_probs = self(tensor([x, y, rz, v]))
            a_dist = Categorical(a_probs)
            w_dist = Categorical(w_probs)
            a_index = a_dist.sample()
            w_index = w_dist.sample()
            self.records.append([
                0,
                a_dist.probs[a_index],
                w_dist.probs[w_index],
                a_dist.log_prob(a_index),
                w_dist.log_prob(w_index)
            ])  # feedback of last episode
            a = self.a_space[a_index].item()
            w = self.w_space[w_index].item()
        else:
            a = w = 0
            self.finish_episode()

        if(done):
            self.v_history = []
            self.a_history = []
            self.w_history = []
            self.r_history = []
        else:
            self.v_history.append(v)
            self.a_history.append(a)
            self.w_history.append(w)
            self.r_history.append(r)
            x_series=list(range(len(self.v_history)))
            self.viz.line(X=x_series, Y=self.v_history,
                          win=self.plot_v, opts=dict(ylabel="velocity"))
            self.viz.line(X=x_series, Y=self.a_history,
                          win=self.plot_a, opts=dict(ylabel="acceleration"))
            self.viz.line(X=x_series, Y=self.w_history,
                          win=self.plot_w, opts=dict(ylabel="front wheel angle"))
            self.viz.line(X=x_series, Y=self.r_history,
                          win=self.plot_r, opts=dict(ylabel="feedback"))
        return a, w

    def finish_episode(self):
        values = []
        for r, ap, wp, lap, lwp in self.records:
            values.append(r)
        values=tensor(values)
        values=standardize(values)
        pstdloss=stdloss=rloss=prloss=0
        for value, record in zip(values,self.records):
            pstdloss += value*record[1]*record[2]
            stdloss+=value
            rloss+=record[0]
            prloss+=record[0]*record[1]*record[2]
        self.optimize(prloss)

        self.l_history.append([pstdloss.item(),prloss.item(),stdloss,rloss])
        self.viz.line(X=list(range(len(self.l_history))), Y=self.l_history,
                      win=self.plot_l, opts=dict(ylabel="loss",legend=["pstd","pr","std","r"]))
        self.viz.line(X=list(range(len(self.records))), Y=tensor(self.records),
                      win=self.plot_value, opts=dict(ylabel="value", width=800,height=200,legend=["r","ap","wp","lap","lwp"]))

        del self.records[:]

        
