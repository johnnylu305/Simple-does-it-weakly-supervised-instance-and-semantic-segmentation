import os
import sys
import numpy as np
import tqdm

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Parser_.parser import make_pair_parser
import voc12_class

# standard output format
SPACE = 35

# tqdm parameter
UNIT_SCALE = True
BAR_FORMAT = '{}{}{}'.format('{l_bar}', '{bar}', '| {n_fmt}/{total_fmt}')


class Maker:

    def __init__(self):
        args = make_pair_parser()
        # get dataset path
        self.dataset_path = args.dataset
        # get training set name
        self.train_name = args.train_set_name
        # get annotation name
        self.ann_dir_name = args.ann_dir_name
        # annotation information
        self.ann_info = np.array([])
        # get train pair name
        self.train_pair_name = args.train_pair_name

    def save_train_pair(self):
        with open(self.dataset_path + '/' + self.train_pair_name,
                  'w') as w, open(self.dataset_path + '/' + self.train_name,
                                  'r') as r:
            # load image name
            for img_name in tqdm.tqdm(
                    r, desc='{:{}}'.format('Save pair name', SPACE),
                    unit_scale=UNIT_SCALE):
                img_name = img_name.rstrip()
                # load annotation
                self.load_annotation(img_name + '.xml')
                # save train pair
                for i, info in enumerate(self.ann_info):
                    if info[0] in voc12_class.voc12_classes:
                        grabcut_name = '{}_{}_{}.png'.format(
                                img_name, i,
                                voc12_class.voc12_classes[info[0]])
                        w.write('{}###{}###{}###{}###{}###{}###{}\n'.format(
                            img_name, grabcut_name, info[2], info[1], info[4],
                            info[3], info[0]))
            r.close()
            w.close()
        print('Save set successful')

    # load annotation
    def load_annotation(self, filename):
        with open(self.dataset_path + '/' + self.ann_dir_name + '/' + filename,
                  'r') as r:
            soup = BeautifulSoup(r, 'xml')
            # get bounding boxes coordinate
            xmins = soup.find_all('xmin')
            ymins = soup.find_all('ymin')
            xmaxs = soup.find_all('xmax')
            ymaxs = soup.find_all('ymax')
            # get class name
            names = soup.find_all('name')
            # extract information
            self.ann_info = np.array([])
            for name, xmin, ymin, xmax, ymax in zip(names, xmins, ymins, xmaxs,
                                                    ymaxs):
                self.ann_info = np.append(self.ann_info, np.array(
                    [name.string, xmin.string, ymin.string, xmax.string,
                     ymax.string]))
            self.ann_info = self.ann_info.reshape(-1, 5)
            r.close()


def main():
    make_pair = Maker()
    make_pair.save_train_pair()


if __name__ == '__main__':
    main()
