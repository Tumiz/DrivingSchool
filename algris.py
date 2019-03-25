from torch import tensor
def normalize(data):# input tensor
    ret = (data-data.mean())/(data.std()+1e-10)
    return ret

def gather(feeds,gamma):#input list
    values=[]
    feedsum=0
    V=0
    for feed in feeds[::-1]:
        V=feed+gamma*V
        values.insert(0,V)
        feedsum+=feed
    return values, feedsum