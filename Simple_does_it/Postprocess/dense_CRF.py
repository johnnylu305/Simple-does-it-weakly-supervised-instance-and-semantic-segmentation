###################################################################################
# Cite from:                                                                      #
#   Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials     #
#   Philipp Krähenbühl and Vladlen Koltun                                         #
#   NIPS 2011                                                                     #
#   Link: https://github.com/lucasb-eyer/pydensecrf                               #
###################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf

class dense_CRF:
    def __init__(self, img, masks):
        self.img = img
        self.masks = masks
        self.width = masks.shape[0]
        self.height = masks.shape[1]
        self.classes = masks.shape[2]
    
    def run_dense_CRF(self):
        # [w, h, class] to [class, w, h]
        U = self.masks.transpose(2, 0, 1).reshape((self.classes, -1))
        U = U.copy(order = 'C')
        # declare width, height, class
        d = dcrf.DenseCRF2D(self.height, self.width, self.classes)
        # set unary potential
        d.setUnaryEnergy(-np.log(U))
        # set pairwise potentials
        d.addPairwiseGaussian(sxy=(3, 3), compat = 3)
        d.addPairwiseBilateral(sxy = 80, srgb = 13, rgbim = self.img, compat = 10)
        # inference with 5 iterations
        Q = d.inference(5)
        # MAP prediction
        map = np.argmax(Q, axis = 0).reshape((self.width, self.height))
        # class-probabilities
        proba = np.array(map)

        return proba





