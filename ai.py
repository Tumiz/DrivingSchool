from torch.optim import Adam
from torch import tensor, arange
from torch.nn import Module, Linear
from torch.nn.functional import relu, softmax, leaky_relu
from torch.distributions import Categorical

import visdom


class Agent(Module):

    def __init__(self):
        super(Agent, self).__init__()
        self.a_space = arange(-1, 1.1, 0.2)
        self.w_space = arange(-0.5, 0.6, 0.1)
        self.s_head = Linear(4, 4*32)
        self.a_head = Linear(4*32, len(self.a_space))
        self.w_head = Linear(4*32, len(self.w_space))
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.records = []
        self.episode = 0

        self.viz = visdom.Visdom()
        self.plot_v = self.viz.line(X=[0], Y=[0])
        self.plot_a = self.viz.line(X=[0], Y=[0])
        self.plot_w = self.viz.line(X=[0], Y=[0])
        self.plot_l = self.viz.line(X=[0], Y=[0])
        self.v_history = []
        self.t_history = []
        self.w_history = []
        self.a_history = []
        self.l_history = []

    def forward(self, x):
        x = leaky_relu(self.s_head(x))
        a_scores = self.a_head(x)
        w_scores = self.w_head(x)
        return softmax(a_scores, dim=0), softmax(w_scores, dim=0)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decision(self, done, x, y, rz, v, R, t):  # return action
        a_probs, w_probs = self(tensor([x, y, rz, v]))
        a_dist = Categorical(a_probs)
        w_dist = Categorical(w_probs)
        a_index = a_dist.sample()
        w_index = w_dist.sample()
        self.records.append((R, a_probs[a_index], w_probs[w_index]))
        a = self.a_space[a_index].item()
        w = self.w_space[w_index].item()

        if(done):
            self.finish_episode()
            self.v_history = [0]
            self.a_history = [0]
            self.w_history = [0]
            self.t_history = [0]
        else:
            self.v_history.append(v)
            self.a_history.append(a)
            self.t_history.append(t)
            self.w_history.append(w)
        self.viz.line(X=self.t_history, Y=self.v_history,
                      win=self.plot_v, opts=dict(ylabel="velocity"))
        self.viz.line(X=self.t_history, Y=self.a_history,
                      win=self.plot_a, opts=dict(ylabel="acceleration"))
        self.viz.line(X=self.t_history, Y=self.w_history,
                      win=self.plot_w, opts=dict(ylabel="front wheel angle"))
        return a, w

    def finish_episode(self):
        R = self.records[-1][0]
        self.records.pop()
        loss = R
        if len(self.records) > 0:
            for r, a_prob, w_prob in self.records[::-1]:
                R = r+0.9*R
                loss = loss+R*a_prob*w_prob
            self.optimize(loss)
            loss=loss.item()
        del self.records[:]

        self.episode+=1
        self.l_history.append(loss)
        self.viz.line(X=list(range(self.episode)),Y=self.l_history,
                      win=self.plot_l, opts=dict(ylabel="loss"))
