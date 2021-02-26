import torch

def get_mode_normal(distribution):
    return distribution.mean

def get_mode_beta(distribution):
    conc0 = distribution.concentration0
    conc1 = distribution.concentration1
    mode = (conc0 - 1) / (conc0 + conc1 - 2)
    return mode

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot

def split_out_continous_rl(out):
    if isinstance(out, dict):
        return out
    else:
        ret_dict = {
            "action_distributions": out[0],
            "vvalues": out[1],
        }
        return ret_dict