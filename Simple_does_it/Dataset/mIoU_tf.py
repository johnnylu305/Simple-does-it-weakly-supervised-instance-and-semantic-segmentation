import numpy as np
import tqdm
import os
import sys
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Parser_.parser import mIoU_parser
from Dataset import voc12_class

# standard output format
SPACE = 35

# tqdm parameter
UNIT_SCALE = True

x = tf.placeholder(tf.float32, shape=(None, None))
y = tf.placeholder(tf.float32, shape=(None, None))


def compute():
    temp = tf.multiply(x, y)
    inter = tf.reduce_sum(temp)
    union = tf.reduce_sum(tf.subtract(tf.add(x, y), temp))
    return inter, union


class IoU:
    def __init__(self, dataset='.'):
        args = mIoU_parser()
        # get dataset path
        self.dataset = args.dataset
        # get set name
        self.set_name = args.set_name
        # get name of ground truth directory
        self.GT_dir_name = args.GT_dir_name
        # get name of prediction directory
        self.Pred_dir_name = args.Pred_dir_name
        # get number of classes
        self.classes = args.classes

    def compute_mIoU(self):
        union_ = np.zeros(self.classes, dtype=int)
        inter_ = np.zeros(self.classes, dtype=int)
        in_, un_ = compute()
        sess = tf.Session()
        with open(self.dataset + '/' + self.set_name, 'r') as r:
            for i, img_name in enumerate(tqdm.tqdm(r,
                                                   desc='{:{}}'.format(
                                                       'Load image', SPACE),
                                                   unit_scale=UNIT_SCALE),
                                         start=1):
                # get image name
                img_name = img_name.rstrip()
                # get Ground Truth
                GT = Image.open(self.dataset + '/' + self.GT_dir_name + '/' +
                                img_name + '.png')
                GT = np.array(GT)
                GT = np.where((GT == 255), 0, GT)
                # get prediction
                Pred = Image.open(self.dataset + '/' + self.Pred_dir_name + '/'
                                  + img_name + '.png')
                Pred = np.array(Pred)
                for i in range(self.classes):
                    target = np.where((GT == i), 1, 0)
                    pred = np.where((Pred == i), 1, 0)
                    in__, un__ = sess.run([in_, un_], feed_dict={x: pred,
                                                                 y: target})
                    union_[i] = union_[i] + un__
                    inter_[i] = inter_[i] + in__
        # compute IoU
        IoU = inter_/union_
        print('{:{}}: {}'.format('Set', SPACE, self.set_name))
        for k, v in zip(voc12_class.voc12_classes.keys(), IoU):
            print('{:{}}: {}'.format(k, SPACE, v))
        # count mIoU
        mIoU = np.mean(IoU)

        print('{:{}}: {}'.format('mIoU', SPACE, mIoU))
        return mIoU


def main():
    result = IoU()
    result.compute_mIoU()


if __name__ == '__main__':
    main()
