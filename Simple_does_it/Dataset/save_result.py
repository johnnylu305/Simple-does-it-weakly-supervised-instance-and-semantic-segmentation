import os
import sys
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Dataset import voc12_color 

class Save:
    def __init__(self, img, masks, img_name, pred_dir_path, pair_dir_path, classes):
        # get segmentation
        self.masks = masks
        # get image
        self.img = img
        # get image name
        self.img_name = img_name
        # get directory for saving prediction 
        self.pred_dir_path = pred_dir_path
        # get directory for 
        self.pair_dir_path = pair_dir_path
        # get classes
        self.classes = classes
    def save(self):
        # save segmentation
        scipy.misc.toimage(self.masks, cmin=0, cmax=255, pal = voc12_color.colors_map, mode = 'P').save(self.pred_dir_path + '/' + self.img_name+'.png')
        # create figure        
        fig = plt.figure()
        
        # convert to inch
        # dpi: dot per inch
        w, h =self.img.shape[1] / float(fig.get_dpi()), self.img.shape[0] / float(fig.get_dpi())
        # set figure size
        fig.set_size_inches(w, h)

        
        for i in range(1, self.classes):
            # get color for mask
            color = voc12_color.colors[i]
            m = self.masks[:, :, np.newaxis]
            # add mask
            for c in range(3):
                self.img[:, :, c] = np.where((m[:, :, 0] == i), self.img[:, :, c] * 0.3 + 0.7 * color[c], self.img[:, :, c])
        # show image
        plt.figimage(self.img)
    
        # save image with grabcut masks
        fig.savefig(self.pair_dir_path + '/' + self.img_name + '.png')
        plt.cla()
        plt.clf()
        plt.close('all')

