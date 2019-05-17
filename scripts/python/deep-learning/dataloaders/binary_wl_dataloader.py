import numpy as np
import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os
import glob
from PIL import Image
import numpy as np
import torch
import os
import sys
import shutil
import math


class BinaryWMLoader(torch_data.Dataset):
    def __init__(self, data_dir, split, num_classes = 1 ):
        self.data_dir = data_dir
        self.split = split
        
        self.num_classes = num_classes
        self.image_list, self.label_list = [], []
        
        self.read_lists()
        
    def read_lists(self):
        data_bins = os.path.join(self.data_dir, self.split)
        assert os.path.exists(data_bins)
        folders = [i for i in next(os.walk(data_bins))[1]]
        for folder in folders:
            for each_file in glob.glob(os.path.join(data_bins , folder) + '/npy/*.npy'):
                self.image_list.append(each_file)
                self.label_list.append(int(folder))
                
    def __getitem__(self, index):
        im = np.load(self.image_list[index])
        data = [im]
        data.append(self.label_list[index])
        return tuple(data)


    def __len__(self):
        return len(self.image_list)