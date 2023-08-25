import argparse
import os
import scipy.io
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

def nmse_tot(input, target):

    output = 10*np.log10(np.linalg.norm(input - target)**2/(np.linalg.norm(target)**2))
    return output


def main(args):


    test_dir = '/nas/home/fmiotello/projects/palette/experiments/test_diff_sfr_230824_164029/results/test/0'
    rooms_dir = '/nas/home/fmiotello/projects/diff_sfr/dataset/test'



    tot_nmse = 0

    for room in os.listdir(rooms_dir):

        # room = '293_d_4.393_9.8538_43.2882_s_0.0045377_4.6913_.mat'

        gt_mat = scipy.io.loadmat(os.path.join(test_dir, 'GT_{}'.format(room)))
        out_mat = scipy.io.loadmat(os.path.join(test_dir, 'Out_{}'.format(room)))

        gt = gt_mat['AbsFrequencyResponse']
        out = out_mat['AbsFrequencyResponse']

        #np.min(out)

        # plt.imshow(out[:,:,0])
        # plt.colorbar()
        # plt.show()

        nmse = nmse_tot(gt, out)
        tot_nmse += nmse
        print('Room_', room.split('_')[0], ' nmse: ', nmse)

print('tot_nmse: ', tot_nmse/500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path of test results')

    ''' parser configs '''
    args = parser.parse_args()

    main(args)