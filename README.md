# Simple-does-it-weakly-supervised-instance-and-semantic-segmentation

## My Environment
### Environment 1
- Operating System:
  - Arch Linux 4.15.15-1  
- CUDA:
  - CUDA V9.0.176 
- CUDNN:
  - CUDNN 7.0.5-2
- GPU:
  - GTX 1070 8G
- Nvidia driver:
  - 390.25
- Python:
  - python 3.6.4
- Tensorflow:
  - tensorflow-gpu 1.5.0
### Environment 2
- Operating System:
  - Ubuntu 16.04  
- CUDA:
  - CUDA V9.0.176 
- CUDNN:
  - CUDNN 7
- GPU:
  - GTX 1060 6G
- Nvidia driver:
  - 390.48
- Python:
  - python 3.5.2
- Tensorflow:
  - tensorflow-gpu 1.6.0
  
## Downloading the VOC12 dataset
- [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [Pascal VOC Dataset Mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

## Setup Dataset
- My directory structure
```
./Simple_does_it/
├── Dataset
│   ├── Annotations
│   ├── CRF_masks
│   ├── CRF_pairs
│   ├── Grabcut_inst
│   ├── Grabcut_pairs
│   ├── JPEGImages
│   ├── Pred_masks
│   ├── Pred_pairs
│   └── Segmentation_label
├── Model
│   ├── Logs
│   └── models
├── Parser_
├── Postprocess
├── Preprocess
└── Util
```
- VOC2012 directory structure
```
VOCtrainval_11-May-2012
└── VOCdevkit
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        │   ├── Action
        │   ├── Layout
        │   ├── Main
        │   └── Segmentation
        ├── JPEGImages
        ├── SegmentationClass
        └── SegmentationObject
```

## Preprocess for training (See Usage for more details)
### Extract annotations from 'Annotations' according 'train.txt'
- This will generate a 'train_pairs.txt' for 'grabcut.py'
```
python Dataset/make_train.py 
```
### Generate label for training by 'grabcut.py'
- Result of grabcut for each bounding box will be stored at 'Grabcut_inst'
- Result of grabcut for each image will be stored at 'Segmentation_label'
- Result of grabcut for each image combing with image and bounding box will be stored at 'Grabcut_pairs'
- Note: If the label has existed at 'Segmentation_label', grabcut.py will skip that image
```
python Preprocess/grabcut.py
```
### Train network
- The event file for tensorboard will be stored at 'Logs'
```
python Model/model.py --is_train 1 --set_name train.txt   
```
## Reference
- [[1] Anna Khoreva, Rodrigo Benenson, Jan Hosang, Matthias Hein, Bernt Schiele. Simple Does It: Weakly Supervised Instance and Semantic Segmentation. CVPR 2017](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/weakly-supervised-learning/simple-does-it-weakly-supervised-instance-and-semantic-segmentation/)
- [[2] philferriere. Weakly Supervised Segmentation with Tensorflow. Implements instance segmentation as described in Simple Does It: Weakly Supervised Instance and Semantic Segmentation, by Khoreva et al. (CVPR 2017).](https://github.com/philferriere/tfwss)
- [[3] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. arXive 2016](https://arxiv.org/abs/1606.00915)
- [[4] Philipp Krähenbühl, Vladlen Koltun. Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials. NIPS 2011](https://arxiv.org/abs/1210.5644)
