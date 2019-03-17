from torch import tensor
def standardize(data):
    ret = (data-data.mean())/(data.std()+1e-10)
    return ret