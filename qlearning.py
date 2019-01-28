import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(2)  # reproducible

EPSILON = 0.8   # greedy police
GAMMA = 0.9    # discount factor
MAX_EPISODES = 100   # maximum episodes
FRESH_TIME = 0.2    # fresh time for one move
TARGET = 7 # where the robot should reach at the end

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(2, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def q(self, S, A):#return Q
        return self(torch.tensor([float(S),A])).detach().numpy()[0]

    def best_action(self, S):
        choices=np.random.uniform(-1,1,20)
        qlist=[]
        for choice in choices:
            qlist.append([self.q(S,choice),choice])
        ret=max(qlist)
        return ret[0],ret[1]#Q,A

    def choose_action(self,S):#return A
        # This is how to choose an action
        if (np.random.uniform() > EPSILON):  # act non-greedy or state-action have no value
            A = random.randint(-1,1)
            Q = self.q(S,A)
            info = "try"
        else:   # act greedy
            Q,A=self.best_action(S)
            info = ""
        return A,Q,info

    def train(self,experience):
        experience=torch.tensor(experience)
        loss_func = nn.MSELoss(reduce = True,size_average = True)
        optimizer=optim.SGD(self.parameters(), lr=0.01)
        input = torch.tensor(experience[:, 0:2]).float()
        target = torch.tensor(experience[:, 2:3]).float()
        loss_min = loss_func(self(input), target)
        counts=0
        while loss_min > 0.05 and counts<1000:
            predict = self(input)
            loss = loss_func(predict, target)
            if(torch.isnan(loss)):
                print("loss is nan")
                break
            if(loss_min - loss < optimizer.param_groups[0]["lr"]):
                loss_min = loss
            else:
                optimizer.param_groups[0]["lr"] *= 0.99
            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            optimizer.step() 
            counts+=1



def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    S_=S+A
    S_=max(0,S_)
    S_=min(10,S_)
    if abs(S_-TARGET)<abs(S-TARGET) or abs(S_-TARGET)<0.1:
        R=1
    else:
        R=0
    return S_, R


def update_env(S, A, Q, info):
    # This is how environment be updated
    env_list = ['-']*(TARGET) + ['T'] + ['-'] * (10-TARGET)   # '---------T' our environment
    env_list[int(round(S))] = 'o'
    interaction = ''.join(env_list)
    print '\r{}'.format(interaction),'%.3f'%S,'%.3f'%A,'%.3f'%Q,info
    time.sleep(FRESH_TIME)


# main part of RL loop
episode=0
driver = Net()
step_counter = 10
experience = []
while episode<=MAX_EPISODES: 
    step_counter=0       
    S = 0
    is_terminated = False
    episode+=1
    
    while not is_terminated:
        Q,A = driver.best_action(S)
        update_env(S, A, Q, "")
        S_, R = get_env_feedback(S, A)  # take action & get next state and reward

        if abs(S_-TARGET)<0.1 and abs(A)<0.1:
            is_terminated = True    # terminate this episode
            interaction = 'Episode %s: total_steps = %s' % (episode, step_counter)
            print('\r{}'.format(interaction))
            print(driver)
            time.sleep(2)
            print('\r                                ')
        record = [S,A,R]
        if record not in experience:
            experience.append(record)
        driver.train(experience)
        S = S_  # move to next state
        step_counter += 1