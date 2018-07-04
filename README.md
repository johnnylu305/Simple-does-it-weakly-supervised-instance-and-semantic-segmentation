# Simple-does-it-weakly-supervised-instance-and-semantic-segmentation
There are five weakly supervised networks in  $\textbf{Simple Does It: Weakly Supervised Instance and Semantic Segmentation }$, by Khoreva et al. (CVPR 2017). Respectively,  $\textbf{aive, Box, Box$^i$, Grabcut+, M$\cap$G+}$. All of them use $\textbf{cheap-to-generate label, bounding box}$, during training and don't need other informations except image during testing.

This repo contains a TensorFlow implementation of $\textbf{Grabcut version}$.

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
- Python package:
  - tqdm, bs4, opencv-python, pydensecrf, cython...
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
- Python package:
  - tqdm, bs4, opencv-python, pydensecrf, cython...
- Tensorflow:
  - tensorflow-gpu 1.6.0
 
## Downloading the VOC12 dataset
- [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [Pascal VOC Dataset Mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

## Setup Dataset
### My directory structure
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
### VOC2012 directory structure
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
- Put annotations in 'Annotations'
```
mv {PATH}/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/* {PATH}/Simple_does_it/Dataset/Annotations/ 
```
- Put images in 'JPEGImages'
```
mv {PATH}/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/* {PATH}/Simple_does_it/Dataset/JPEGImages/
```
- Put train.txt in 'Dataset'
```
mv {PATH}/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt {PATH}/Simple_does_it/Dataset/  
```
- Put val.txt in 'Dataset'
```
mv {PATH}/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt {PATH}/Simple_does_it/Dataset/  
```

## Training (See Usage for more details)
### Download pretrain vgg16
- [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/slim)
  - Put vgg_16.ckpt in 'models'
  
### Extract annotations from 'Annotations' according to 'train.txt'
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

## Testing (See Usage for more details)
### Test network
- Result will be stored at 'Pred_masks'
- Result combing with image will be stored at 'Pred_pairs'
- Result after dense CRF will be stored at 'CRF_masks'
- Result after dense CRF combing with image will be stored at 'CRF_pairs'
```
python Model/model.py 
```

## Usage
### Parser_/parser.py
- Parse the command line argument
### Util/divied.py
- Generating train.txt and test.txt according to 'JPEGImages'
- Not necessary
```
usage: divied.py [-h] [--dataset DATASET] [--img_dir_name IMG_DIR_NAME]
                 [--train_set_ratio TRAIN_SET_RATIO]
                 [--train_set_name TRAIN_SET_NAME]
                 [--test_set_name TEST_SET_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     path to dataset (default: Util/../Parser_/../Dataset)
  --img_dir_name IMG_DIR_NAME
                        name for image directory (default: JPEGImages)
  --train_set_ratio TRAIN_SET_RATIO
                        ratio for training set, [0,10] (default: 7)
  --train_set_name TRAIN_SET_NAME
                        name for training set (default: train.txt)
  --test_set_name TEST_SET_NAME
                        name for testing set (default: val.txt)
```
### Dataset/make_train.py
- Extract annotations from 'Annotations' according to 'train.txt'
- Content: {image name}###{image name + num + class + .png}###{bbox ymin}###{bbox xmin}###{bbox ymax}###{bbox xmax}###{class}
- Example: 2011_003038###2011_003038_3_15.png###115###1###233###136###person
```
usage: make_train.py [-h] [--dataset DATASET]
                     [--train_set_name TRAIN_SET_NAME]
                     [--ann_dir_name ANN_DIR_NAME]
                     [--train_pair_name TRAIN_PAIR_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     path to dataset (default:
                        Dataset/../Parser_/../Dataset)
  --train_set_name TRAIN_SET_NAME
                        name for training set (default: train.txt)
  --ann_dir_name ANN_DIR_NAME
                        name for annotation directory (default: Annotations)
  --train_pair_name TRAIN_PAIR_NAME
                        name for training pair (default: train_pairs.txt)
```
### Preprocess/grabcut.py
- Grabcut a traditional computer vision method
- Input bounding box and image then generating label for training
```
usage: grabcut.py [-h] [--dataset DATASET] [--img_dir_name IMG_DIR_NAME]
                  [--train_pair_name TRAIN_PAIR_NAME]
                  [--grabcut_dir_name GRABCUT_DIR_NAME]
                  [--img_grabcuts_dir IMG_GRABCUTS_DIR]
                  [--pool_size POOL_SIZE] [--grabcut_iter GRABCUT_ITER]
                  [--label_dir_name LABEL_DIR_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     path to dataset (default:
                        Preprocess/../Parser_/../Dataset)
  --img_dir_name IMG_DIR_NAME
                        name for image directory (default: JPEGImages)
  --train_pair_name TRAIN_PAIR_NAME
                        name for training pair (default: train_pairs.txt)
  --grabcut_dir_name GRABCUT_DIR_NAME
                        name for grabcut directory (default: Grabcut_inst)
  --img_grabcuts_dir IMG_GRABCUTS_DIR
                        name for image with grabcuts directory (default:
                        Grabcut_pairs)
  --pool_size POOL_SIZE
                        pool for multiprocess (default: 4)
  --grabcut_iter GRABCUT_ITER
                        grabcut iteration (default: 5)
  --label_dir_name LABEL_DIR_NAME
                        name for label directory (default: Segmentation_label)
```
### Model/model.py
- Deeplab-Largefov
```
usage: model.py [-h] [--dataset DATASET] [--set_name SET_NAME]
                [--label_dir_name LABEL_DIR_NAME]
                [--img_dir_name IMG_DIR_NAME] [--classes CLASSES]
                [--batch_size BATCH_SIZE] [--epoch EPOCH]
                [--learning_rate LEARNING_RATE] [--momentum MOMENTUM]
                [--keep_prob KEEP_PROB] [--is_train IS_TRAIN]
                [--save_step SAVE_STEP] [--pred_dir_name PRED_DIR_NAME]
                [--pair_dir_name PAIR_DIR_NAME] [--crf_dir_name CRF_DIR_NAME]
                [--crf_pair_dir_name CRF_PAIR_DIR_NAME] [--width WIDTH]
                [--height HEIGHT] [--restore_target RESTORE_TARGET]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     path to dataset (default: Model/../Parser_/../Dataset)
  --set_name SET_NAME   name for set (default: val.txt)
  --label_dir_name LABEL_DIR_NAME
                        name for label directory (default: Segmentation_label)
  --img_dir_name IMG_DIR_NAME
                        name for image directory (default: JPEGImages)
  --classes CLASSES     number of classes for segmentation (default: 21)
  --batch_size BATCH_SIZE
                        batch size for training (default: 3)
  --epoch EPOCH         epoch for training (default: 30000)
  --learning_rate LEARNING_RATE
                        learning rate for training (default: 0.0005)
  --momentum MOMENTUM   momentum for optimizer (default: 0.9)
  --keep_prob KEEP_PROB
                        probability for dropout (default: 1)
  --is_train IS_TRAIN   training or testing [1 = True / 0 = False] (default:
                        0)
  --save_step SAVE_STEP
                        step for saving weight (default: 10)
  --pred_dir_name PRED_DIR_NAME
                        name for prediction masks directory (default:
                        Pred_masks)
  --pair_dir_name PAIR_DIR_NAME
                        name for pairs directory (default: Pred_pairs)
  --crf_dir_name CRF_DIR_NAME
                        name for crf prediction masks directory (default:
                        CRF_masks)
  --crf_pair_dir_name CRF_PAIR_DIR_NAME
                        name for crf pairs directory (default: CRF_pairs)
  --width WIDTH         width for resize (default: 400)
  --height HEIGHT       height for resize (default: 400)
  --restore_target RESTORE_TARGET
                        target for restore (default: 100)
```
### Dataset/load.py   
- Loading data for training / testing according to train.txt / val.txt
### Dataset/save_result.py  
- Save result during testing
### Dataset/voc12_class.py  
- Map the class to grayscale value
### Dataset/voc12_color.py  
- Map the grayscale value to RGB
###  Postprocess/dense_CRF.py 
- Dense CRF a machine learning method
- Refine the result

## Future plan
- Tune hyperparameters
- Multiscale Combinatorial Grouping
- Grabcut+

## Reference
- [[1] Anna Khoreva, Rodrigo Benenson, Jan Hosang, Matthias Hein, Bernt Schiele. Simple Does It: Weakly Supervised Instance and Semantic Segmentation. CVPR 2017](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/weakly-supervised-learning/simple-does-it-weakly-supervised-instance-and-semantic-segmentation/)
- [[2] philferriere. Weakly Supervised Segmentation with Tensorflow. Implements instance segmentation as described in Simple Does It: Weakly Supervised Instance and Semantic Segmentation, by Khoreva et al. (CVPR 2017).](https://github.com/philferriere/tfwss)
- [[3] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. arXive 2016](https://arxiv.org/abs/1606.00915)
- [[4] Philipp Krähenbühl, Vladlen Koltun. Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials. NIPS 2011](https://arxiv.org/abs/1210.5644)
