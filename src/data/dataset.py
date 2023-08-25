import torch.utils.data as data
from torchvision import transforms
import scipy.io
import os
import torch
import numpy as np
import random

from .util.mask import get_sfr_mask

MAT_EXTENSION = '.mat'


def is_mat_file(filename):
    return filename.endswith(MAT_EXTENSION)


def make_dataset(dir):
    freq_responses = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                freq_responses.append(path)
    return freq_responses


def mat_loader(path, freq):
    frequencies = np.asarray(freq)
    mat = scipy.io.loadmat(path)
    f_response = mat['AbsFrequencyResponse']
    # f_response = np.transpose(f_response, (1, 0, 2))
    soundfield = f_response[:, :, frequencies]
    return soundfield


def get_frequencies():
    freqs_path = 'data/util/frequencies.txt'
    with open(freqs_path) as f:
        freqs = [[int(freq) for freq in line.strip().split(' ')] for line in f.readlines()][0]
    return freqs


def scale(soundfield):
    for i in range(soundfield.shape[-1]):
        max_abs_freq_i = np.max(soundfield[:,:,i])
        soundfield[:,:,i] = soundfield[:,:,i]/max_abs_freq_i
    return soundfield



class SoundFieldReconstructionDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[32, 32], loader=mat_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        freqs = get_frequencies()
        img = self.tfs(scale(self.loader(path, freqs))).to(torch.float32)
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.rand(len(freqs), self.image_size[0], self.image_size[1]) #mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'sfr_mask':
            mask = get_sfr_mask(self.image_size, random.randint(64, 256))
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)