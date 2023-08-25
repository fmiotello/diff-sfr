import numpy as np
import random


def get_sfr_mask(img_shape, n_mics, dtype='uint8'):
    # '1' indicates the hole and '0' indicates the valid regions.

    assert n_mics < np.prod(img_shape)
    mask = np.ones(img_shape, dtype=dtype)

    flat_idx = random.sample(range(np.prod(img_shape)), n_mics)
    (row, col) = np.unravel_index(flat_idx, img_shape)

    for (i, j) in zip(row, col):
        mask[i, j] = 0

    mask = np.expand_dims(mask, axis=2)

    return mask