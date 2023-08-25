import torch
from torch import nn
import torch.utils.data


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def nmse_tot(input, target):
    with torch.no_grad():
        output = 10*torch.log10(torch.linalg.norm(input - target)**2/(torch.linalg.norm(target)**2))
    return output


def nmse_freq_0(input, target):
    with torch.no_grad():
        output = 10*torch.log10(torch.linalg.norm(input[:,:,0] - target[:,:,0])**2/(torch.linalg.norm(target[:,:,0])**2))
    return output


def nmse_freq_9(input, target):
    with torch.no_grad():
        output = 10*torch.log10(torch.linalg.norm(input[:,:,9] - target[:,:,9])**2/(torch.linalg.norm(target[:,:,9])**2))
    return output


def nmse_freq_19(input, target):
    with torch.no_grad():
        output = 10*torch.log10(torch.linalg.norm(input[:,:,19] - target[:,:,19])**2/(torch.linalg.norm(target[:,:,19])**2))
    return output


def nmse_freq_29(input, target):
    with torch.no_grad():
        output = 10*torch.log10(torch.linalg.norm(input[:,:,29] - target[:,:,29])**2/(torch.linalg.norm(target[:,:,29])**2))
    return output