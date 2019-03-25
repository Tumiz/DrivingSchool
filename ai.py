from torch.optim import Adam
from torch import tensor, arange, stack
from torch.nn import Module, Linear
from torch.nn.functional import softmax, elu
from torch.distributions import Categorical
from algris import normalize

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
        self.f_history = []
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

    def decision(self, done, x, y, rz, v, feed, t):  # return action
        if len(self.records):
            self.records[-1][0] = feed
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
                a_dist.log_prob(a_index),  # entropy, negtive
                w_dist.log_prob(w_index)
            ])  # record of last episode
            a = self.a_space[a_index].item()
            w = self.w_space[w_index].item()
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
        values = []
        for feed, ap, wp, lap, lwp in self.records:
            values.append(feed)
        values = tensor(values)
        values = normalize(values)
        psloss = lsloss = prloss = lrloss = []
        rloss = 0
        for value, record in zip(values, self.records):
            psloss.append(value*record[1]*record[2])  # value*p(a)*p(w)
            # log(1/p(a,w))=-log(p(a)*p(w))=-log(p(a))-log(p(w))
            lsloss.append(value*(-record[3]-record[4]))
            prloss.append(record[0]*record[1]*record[2])  # feed*p(a)*p(w)
            lrloss.append(record[0]*(-record[3]-record[4]))  # feed*log(1/p(a,w))
            rloss += record[0]  # sum(feed0,feed1,...,feedt), no gradient
        loss=stack(prloss).sum()
        self.optimize(loss)

        self.l_history.append(
            [loss.item(), rloss])
        self.viz.line(X=list(range(len(self.l_history))), Y=self.l_history,
                      win=self.plot_l, opts=dict(ylabel="loss", legend=["l", "r"]))
        self.viz.line(X=list(range(len(self.records))), Y=tensor(self.records),
                      win=self.plot_value, opts=dict(ylabel="record", width=1000, height=400, legend=["feed", "ap", "wp", "lap", "lwp"]))

        del self.records[:]
