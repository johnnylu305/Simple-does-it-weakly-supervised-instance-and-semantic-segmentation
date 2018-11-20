import numpy as np
import scipy.misc
import tqdm
import os
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Parser_.parser import boxi_parser
from Dataset import voc12_color
from Dataset import voc12_class

# tqdm parameter
UNIT_SCALE = True
BAR_FORMAT = '{}{}{}'.format('{l_bar}', '{bar}', '| {n_fmt}/{total_fmt}')

# standard output format
SPACE = 35


def create(set_, ann_path, label_path):
    # load set
    with open(set_, 'r') as r:
        for f in tqdm.tqdm(r, desc='{:{}}'.format('Create boxi label', SPACE),
                           unit_scale=UNIT_SCALE):
            f = f.rstrip()
            # get label
            save(f, ann_path, label_path)


def save(file_, ann_path, label_path):
    with open(ann_path+'/'+file_+'.xml', 'r') as r:
        soup = BeautifulSoup(r, 'xml')
        # get image size
        size = soup.find('size')
        width = int(size.find('width').string)
        height = int(size.find('height').string)
        # create mask
        mask = np.zeros((height, width), np.uint8)
        # annotations
        anns = []
        # get onjects
        objects = soup.find_all(['object'])
        # get object
        for object_ in objects:
            # get class
            name = object_.find('name').string
            if name not in voc12_class.voc12_classes:
                continue
            class_ = voc12_class.voc12_classes[name]
            # get bounding box
            xmin = int(object_.find('xmin').string)
            xmax = int(object_.find('xmax').string)
            ymin = int(object_.find('ymin').string)
            ymax = int(object_.find('ymax').string)
            # compute width and height
            width = xmax-xmin
            height = ymax-ymin
            # compute area
            area = width*height
            # compute in width and height
            in_xmin = int(xmin+width*0.4)
            in_ymin = int(ymin+height*0.4)
            in_xmax = int(xmax-width*0.4)
            in_ymax = int(ymax-height*0.4)
            # save annotation
            anns.append([area, xmin, ymin, xmax, ymax, in_xmin, in_ymin,
                        in_xmax, in_ymax, class_])
        anns.sort(reverse=True)
        for ann in anns:
            # ignore label
            mask[ann[2]:ann[4], ann[1]:ann[3]] = 22
            # class label
            mask[ann[6]:ann[8], ann[5]:ann[7]] = ann[-1]
        mask = scipy.misc.toimage(mask, cmin=0, cmax=255,
                                  pal=voc12_color.colors_map, mode='P')
        mask.save(label_path+'/'+file_+'.png')


def main():
    args = boxi_parser()
    # get dataset path
    dataset_path = args.dataset
    # get annotations directory path
    ann_path = dataset_path + '/' + args.ann_dir_name
    # get set name
    set_ = dataset_path + '/' + args.set_name
    # get label directory path
    label_path = dataset_path + '/' + args.label_dir_name
    # create boxi label
    create(set_, ann_path, label_path)


if __name__ == '__main__':
    main()
