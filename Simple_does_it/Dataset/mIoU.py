import numpy as np
import tqdm
import matplotlib as mlp
import os
import sys
from PIL import Image

mlp.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Parser_.parser import mIoU_parser
from Dataset import voc12_class

# standard output format
SPACE = 35

# tqdm parameter
UNIT_SCALE = True


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

        self.union = np.zeros(self.classes, dtype=int)
        self.intersection = np.zeros(self.classes, dtype=int)

    def count_mIoU(self):
        with open(self.dataset + '/' + self.set_name, 'r') as r:
            for i, img_name in enumerate(tqdm.tqdm(
                r, desc='{:{}}'.format('Load image', SPACE),
                    unit_scale=UNIT_SCALE), start=1):
                # get image name
                img_name = img_name.rstrip()
                # get Ground Truth
                GT = Image.open(self.dataset + '/' + self.GT_dir_name + '/' +
                                img_name + '.png')
                # GT = GT.resize((513, 513), Image.NEAREST)
                GT = np.array(GT)
                GT = np.where((GT == 255), 0, GT)
                height, width = GT.shape
                # get prediction
                Pred = Image.open(self.dataset + '/' + self.Pred_dir_name +
                                  '/' + img_name + '.png')
                Pred = np.array(Pred)
                Pred = np.where((Pred == 255), 0, Pred)
                # count union and intersection
                for j in range(height):
                    for k in range(width):
                        if Pred[j][k] == GT[j][k]:
                            # union
                            self.union[Pred[j][k]] += 1
                            # intersection
                            self.intersection[Pred[j][k]] += 1
                        else:
                            # union
                            self.union[Pred[j][k]] += 1
                            self.union[GT[j][k]] += 1
            r.close()
        # count IoU
        IoU = np.divide(self.intersection, self.union,
                        out=np.ones(self.classes), where=self.union != 0)
        print('{:{}}: {}'.format('Set', SPACE, self.set_name))
        for x, y in zip(voc12_class.voc12_classes.keys(), IoU):
            print('{:{}}: {}'.format(x, SPACE, y))
        # count mIoU
        mIoU = np.mean(IoU)

        print('{:{}}: {}'.format('mIoU', SPACE, mIoU))
        return mIoU


def main():
    result = IoU()
    result.count_mIoU()


if __name__ == '__main__':
    main()
