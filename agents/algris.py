from torch import tensor
from visdom import Visdom
from torch import tensor, arange
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
    return values

def plot(x,y):
    viz.line(X=x,Y=y)
