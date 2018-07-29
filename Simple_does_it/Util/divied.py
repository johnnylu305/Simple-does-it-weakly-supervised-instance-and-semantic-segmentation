import os
import sys
import numpy as np
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Parser_.parser import divide_parser

# standard output format
SPACE = 35

# tqdm parameter
UNIT_SCALE = True
BAR_FORMAT = '{}{}{}'.format('{l_bar}', '{bar}', '| {n_fmt}/{total_fmt}')


class Divider:

    def __init__(self, args = divide_parser()):
        # get dataset path
        self.dataset_path = args.dataset
        # get image directory path
        self.img_dir_path = self.dataset_path + '/' + args.img_dir_name
        # training set ratio
        self.train_ratio = args.train_set_ratio
        # testing set ratio
        self.test_ratio = args.test_set_ratio
        # training set name
        self.train_name = args.train_set_name
        # testing set name
        self.test_name = args.test_set_name
        # image names
        self.img_names = np.array([])

    # load image names from image directory
    def load_image(self):
        # load image names
        for filename in tqdm.tqdm(os.listdir(self.img_dir_path), desc='{:{}}'.format('Load image names', SPACE), unit_scale = UNIT_SCALE, bar_format = BAR_FORMAT):
            filename = filename[:filename.rfind('.')]
            self.img_names = np.append(self.img_names, filename)
        # get amount of images 
        self.img_num = self.img_names.size
        print ('{:{}}: {}'.format('Samples', SPACE, self.img_num))
        # shuffle images
        np.random.shuffle(self.img_names)
    
    # divide into training set and testing set
    def divide(self):
        # get training set size 
        train_end = int(self.train_ratio*(self.img_num / 10.0))
        # load training set
        train_set = self.img_names[:train_end]
        # load testing set
        test_set = self.img_names[train_end:]
        # print training set size
        print ('{:{}}: {}'.format('Training set size', SPACE, train_set.size))
        # print testing set size
        print ('{:{}}: {}'.format('Testing set size', SPACE, test_set.size))
        if self.train_ratio != 0:
            # save training set 
            self.save_set(train_set, self.train_name)
        if self.test_ratio != 0:
            # save testing set
            self.save_set(test_set, self.test_name)

    # save training set or testing set
    def save_set(self,set_, filename):
        with open(self.dataset_path + '/' + filename, 'w') as w:
            for name in tqdm.tqdm(set_,desc = '{:{}}'.format('Saving testing set', SPACE), unit_scale = UNIT_SCALE, bar_format = BAR_FORMAT):
                w.write(name + '\n')
            w.close()


if __name__ == '__main__':
   divider  = Divider()
   divider.load_image()
   divider.divide()

        

