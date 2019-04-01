from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh
from torch.nn import Module, Linear
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from algris import normalize, gather

import visdom


class Policy(Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.s_head = Linear(2, 2*16)  # accept state
        # a normal distribution represents output of 'a', mu,sigma
        self.a_head = Linear(2*16, 2)

    def forward(self, x):
        x1 = elu(self.s_head(x))
        return self.a_head(x1)
        


class Agent(Module):

    def __init__(self):
        super(Agent, self).__init__()
        self.policy = Policy()
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.feeds = []  # records of returns
        self.a_logprobs = []

    def select_action(self,state):
        a_mu,a_sigma=self.policy(state)
        self.state_actions.append([state,a_mu,a_sigma])
        a_sigma = softplus(a_sigma)
        a_dist = Normal(a_mu, a_sigma)
        a = a_dist.sample()
        a_logprob = a_dist.log_prob(a)
        return a

    def finish_episode(self):
        values = gather(self.feeds, 0.99)
        tvalues = tensor(values)
        nvalues = normalize(tvalues)
        losses = []
        for nvalue, a_logprob in zip(nvalues, self.a_logprobs, self.w_logprobs):
            # log(1/p(a,w))=-log(p(a)*p(w))=-log(p(a))-log(p(w))
            losses.append(nvalue*(-a_logprob-w_logprob))
        loss = stack(losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.feeds = []
        self.a_logprobs = []
        
agent=Agent()
running_reward = 10
for i_episode in count(1):
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000):  # Don't infinite loop while learning
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        agent.feeds.append(reward)
        ep_reward += reward
        if done:
            break

    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
              i_episode, ep_reward, running_reward))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break
