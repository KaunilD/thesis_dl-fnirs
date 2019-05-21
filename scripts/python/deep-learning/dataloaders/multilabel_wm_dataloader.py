import os
import glob
import numpy as np
import sys
import shutil
import math
import numpy as np

import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MultilabelWMLoader(torch_data.Dataset):
    def __init__(self, data_dir, split, num_classes = 1 ):
        self.data_dir = data_dir
        self.split = split

        self.num_classes = num_classes
        self.image_list, self.label_list = [], []

        self.read_lists()

    def read_lists(self):
        data_bins = os.path.join(self.data_dir, self.split)
        assert os.path.exists(data_bins)
        for each_file in glob.glob(data_bins + '\\' + '*.npy'):
            data = np.load(each_file)
            self.image_list.append(data[0])
            self.label_list.append(data[1])

    def __getitem__(self, index):
        return tuple((self.image_list[index], self.label_list[index]))

    def __len__(self):
        return len(self.image_list)
