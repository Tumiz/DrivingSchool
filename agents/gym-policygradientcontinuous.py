from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh, from_numpy
from torch.nn import Module, Linear
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from algris import normalize, gather
from itertools import count
from numpy import array
import argparse
import gym

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('MountainCarContinuous-v0')
env.seed(args.seed)

class Policy(Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.s_head = Linear(2, 2*16)  # accept state
        # a normal distribution represents output of 'a', mu,sigma
        self.a_head = Linear(2*16, 2)

    def forward(self, x):
        print("-------X",x)
        x1 = elu(self.s_head(x))
        return self.a_head(x1)
        
state_=[]
class Agent(Module):

    def __init__(self):
        super(Agent, self).__init__()
        self.policy = Policy()
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.feeds = []  # records of returns
        self.a_logprobs = []

    def select_action(self,state):
        global state_
        state_=tensor(state)
        a_mu,a_sigma=self.policy(tensor(state))
        a_sigma = softplus(a_sigma)
        a_dist = Normal(a_mu, a_sigma)
        a = a_dist.sample()
        a_logprob = a_dist.log_prob(a)
        self.a_logprobs.append(a_logprob)
        return a.numpy()

    def finish_episode(self):
        values = gather(self.feeds, 0.99)
        tvalues = tensor(values)
        nvalues = normalize(tvalues)
        losses = []
        for nvalue, a_logprob in zip(nvalues, self.a_logprobs):
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
        print(action)
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
