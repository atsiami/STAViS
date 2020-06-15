import numpy as np
import torch
import torch.nn.functional as F

def logit(x):
    return np.log(x/(1-x+1e-08)+1e-08)


def sigmoid_np(x):
    return 1/(1+np.exp(-x))


def cc_score(x, y, weights, batch_average=False, reduce=True):

    x=x.squeeze(1)
    x = F.sigmoid(x)
    y=y.squeeze(1)
    mean_x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    mean_y = torch.mean(torch.mean(y, 1, keepdim=True), 2, keepdim=True)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(torch.sum(torch.mul(xm,ym), 1, keepdim=True), 2, keepdim=True)
    r_den_x = torch.sum(torch.sum(torch.mul(xm, xm), 1, keepdim=True), 2, keepdim=True)
    r_den_y = torch.sum(torch.sum(torch.mul(ym, ym), 1, keepdim=True), 2, keepdim=True) + np.asscalar(np.finfo(np.float32).eps)
    r_val = torch.div(r_num, torch.sqrt(torch.mul(r_den_x,r_den_y)))
    r_val = torch.mul(r_val.squeeze(),weights)
    if batch_average:
        r_val = -torch.sum(r_val) / torch.sum(weights)
    else:
        if reduce:
            r_val = -torch.sum(r_val)
        else:
            r_val = -r_val
    return r_val

def nss_score(x, y, weights, batch_average=False, reduce=True):

    x=x.squeeze(1)
    x = F.sigmoid(x)
    y=y.squeeze(1)
    y=torch.gt(y, 0.0).float()

    mean_x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    std_x = torch.sqrt(torch.mean(torch.mean(torch.pow(torch.sub(x, mean_x), 2), 1, keepdim=True), 2, keepdim=True))
    x_norm = torch.div(torch.sub(x, mean_x), std_x)
    r_num = torch.sum(torch.sum(torch.mul(x_norm, y), 1, keepdim=True), 2, keepdim=True)
    r_den = torch.sum(torch.sum(y, 1, keepdim=True), 2, keepdim=True)
    r_val = torch.div(r_num, r_den + np.asscalar(np.finfo(np.float32).eps))
    r_val = torch.mul(r_val.squeeze(), weights)
    if batch_average:
        r_val = -torch.sum(r_val) / torch.sum(weights)
    else:
        if reduce:
            r_val = -torch.sum(r_val)
        else:
            r_val = -r_val
    return r_val

def batch_image_sum(x):
    x = torch.sum(torch.sum(x, 1, keepdim=True), 2, keepdim=True)
    return x

def batch_image_mean(x):
    x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    return x
    
def cross_entropy_loss(output, label, weights, batch_average=False, reduce=True):

    batch_size = output.size(0)
    output = output.view(batch_size, -1)
    label = label.view(batch_size, -1)

    label = label / 255
    final_loss = F.binary_cross_entropy_with_logits(output, label, reduce=False).sum(1)
    final_loss = final_loss * weights

    if reduce:
        final_loss = torch.sum(final_loss)
    if batch_average:
        final_loss /= torch.sum(weights)

    return final_loss