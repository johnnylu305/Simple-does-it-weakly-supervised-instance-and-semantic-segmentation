import os
import sys
import tqdm
import cv2
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import scipy.misc
from multiprocessing import Pool

sys.path.insert(0,os.path.join(os.path.dirname(__file__), '..'))

from Parser_.parser import grabcut_parser
from Dataset import voc12_color
from Dataset import voc12_class

# standard output format
SPACE = 35

# tqdm parameter
UNIT_SCALE = True

class Grabcut:
    
    def __init__(self): 
        args = grabcut_parser()
        # get dataset path
        self.dataset_path = args.dataset
        # get image directory path
        self.img_dir_path = self.dataset_path + '/' + args.img_dir_name
        # get train pair name
        self.train_pair_name = args.train_pair_name
        # get grabcut direcrory name
        self.grabcut_dir_name = args.grabcut_dir_name
        # get image with grabcuts name
        self.img_grabcuts_dir = args.img_grabcuts_dir
        # get pool size
        self.pool_size = args.pool_size
        # get grabcut iteration
        self.grabcut_iter = args.grabcut_iter
        # get label directory name
        self.label_dir_name = args.label_dir_name
        # get annotations
        self.anns = {}
        # ungrabcut image amount
        self.img_num = 0
      

    def load_annotation(self):
        # record grabcut or not
        table = {}
        with open(self.dataset_path + '/' + self.train_pair_name, 'r') as r:
            for i, ann in enumerate(tqdm.tqdm(r, desc='{:{}}'.format('Load annotations', SPACE), unit_scale = UNIT_SCALE), start = 1):
                # split annotation
                ann = ann.rstrip().split('###')
                # initial dict for key
                if ann[0] not in self.anns:
                    self.anns[ann[0]] = []
                # initial dict for key
                if ann[0] not in table:
                    table[ann[0]] = False
                # check grabcut or not
                if table[ann[0]] or not os.path.isfile(self.dataset_path + '/' + self.grabcut_dir_name + '/' + ann[1]):
                    table[ann[0]] = True
                # load annotation
                self.anns[ann[0]].append(ann)
        r.close()
        # leave ungrabcut item
        for key in table:
            if table[key]:
                self.img_num += len(self.anns[key])
            else:
                self.anns.pop(key, None)
        try:
            print ('{:{}}: {}'.format('Total images', SPACE, i))
            print ('{:{}}: {}'.format('Ungrabcut  images', SPACE, self.img_num))
        except UnboundLocalError:
            print ('{:{}}: {}'.format('Total images', SPACE, 0))
            print ('{:{}}: {}'.format('Ungrabcut  images', SPACE, self.img_num))

    def run_grabcut(self):
        # generate pool for multiprocessing
        p = Pool(self.pool_size)
        # run grabcut by multiprocessing
        for _ in tqdm.tqdm(p.imap_unordered(self.grabcut, self.anns), total=len(self.anns)): 
            pass
        p.close()
        p.join()

    def grabcut(self, key):
        masks = []
        for i, ann in enumerate(self.anns[key], start = 1):
            # get annotation
            img_name, grab_img_name, miny, minx, maxy, maxx, class_ = ann
            miny, minx, maxy, maxx = self.str_to_int(miny), self.str_to_int(minx), self.str_to_int(maxy), self.str_to_int(maxx)    
            # load image
            img = cv2.imread(self.img_dir_path + '/' + img_name + '.jpg')
            # grabcut parameter
            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            width = maxx - minx
            height = maxy - miny
            rect = (minx, miny, width, height)
            # run grabcut
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, self.grabcut_iter, cv2.GC_INIT_WITH_RECT)
            # to binary mask
            img_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            # if mask2 no forground
            # reset mask2
            if np.sum(img_mask) == 0:
                img_mask = np.where((mask == 0), 0, 1).astype('uint8')
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # boundingbox to binary mask 
            bbox = np.zeros((img.shape[0], img.shape[1]))
            bbox[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1

            # count IOU
            combine = bbox + img_mask
            intersection = np.where((combine == 2), 1, 0).astype('float')
            union = np.where((combine == 0), 0, 1).astype('float')
            IOU = np.sum(intersection) / np.sum(union) 
            # if IOU less than 15%
            # reset img_mask to bbox
            if IOU < 0.15:
                img_mask = bbox

            masks.append([img_mask, grab_img_name, rect])
        
        # sort by foreground size
        masks.sort(key = lambda mask: np.sum(mask[0]), reverse = True)            
        
        for j in range(i):
            for k in range(j + 1,i):
                masks[j][0] = masks[j][0] - masks[k][0]
            masks[j][0] = np.where((masks[j][0] == 1), 1, 0).astype('uint8')
            # get class
            grab_img_name = masks[j][1]
            class_ = grab_img_name.split('_')[-1]
            class_ = int(class_[:class_.rfind('.')])
            # set class
            masks[j][0] = np.where((masks[j][0] == 1), class_, 0).astype('uint8')
            # save mask
            scipy.misc.toimage(masks[j][0], cmin = 0, cmax = 255, pal = voc12_color.colors_map, mode = 'P').save(self.dataset_path + '/' + self.grabcut_dir_name + '/' + masks[j][1]) 

        # merge masks 
        mask = np.zeros(mask[0][0].shape)
        for m in masks:
            mask = mask + m[0]
        # save merged mask
        scipy.misc.toimage(mask, cmin = 0, cmax = 255, pal = voc12_color.colors_map, mode = 'P').save(self.dataset_path + '/' + self.label_dir_name + '/' + img_name + '.png')
        # create figure        
        fig = plt.figure()
        
        # convert to inch
        # dpi: dot per inch
        w, h =img.shape[1] / float(fig.get_dpi()), img.shape[0] / float(fig.get_dpi())
        # set figure size
        fig.set_size_inches(w, h)

        
        for m in masks:
            rect = m[2]
            m = m[0]
            # get color for mask
            color = voc12_color.colors[np.amax(m)]
            m = m[:, :, np.newaxis]
            # add mask
            for c in range(3):
                img[:, :, c] = np.where((m[:, :, 0] != 0),img[:, :, c] * 0.2 + 0.8 * color[c], img[:, :, c])
            # compute coordinates
            left = rect[0] / img.shape[1]
            bottom = 1 - (rect[1] + rect[3]) / img.shape[0]
            width = (rect[0] + rect[2]) / img.shape[1] - left
            height = 1 - (rect[1]) / img.shape[0] - bottom
            # set bounding box
            ax = fig.add_axes([left, bottom, width, height])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.patch.set_fill(False)
            ax.patch.set_linewidth(5)
            ax.patch.set_color('b')
            # show image
            plt.figimage(img)
        
        # save image with grabcut masks
        fig.savefig(self.dataset_path + '/' + self.img_grabcuts_dir + '/' + img_name + '.png')
        plt.cla()
        plt.clf()
        plt.close()        
    @staticmethod    
    def str_to_int(str_):
        try: return int(str_)
        # Some bounding box coordinates in VOC2012 is float
        # Such as 2011_006777.xml and 2011_003353.xml
        except ValueError: return int(eval(str_))
                

def main():
    grabcut_ = Grabcut()
    grabcut_.load_annotation()
    grabcut_.run_grabcut()

if __name__=='__main__':
    main()
