import argparse
import os

basedir = os.path.join(os.path.dirname(__file__), '..')


# standard output format
SPACE = 35


def divide_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # path to dataset
    # dafault: ../Dataset
    parser.add_argument('--dataset', type=str, default=basedir + '/Dataset',
                        help='path to dataset')
    # name for image directory
    # default: JPEGimages
    parser.add_argument('--img_dir_name', type=str, default='JPEGImages',
                        help='name for image directory')
    # ratio for training set
    # default: 7
    parser.add_argument('--train_set_ratio', type=int,
                        default=7, help='ratio for training set, [0,10]')
    # name for training set
    # default: train.txt
    parser.add_argument('--train_set_name', type=str, default='train.txt',
                        help='name for training set')
    # name for testing set
    # default: test.txt
    parser.add_argument('--test_set_name', type=str, default='test.txt',
                        help='name for testing set')

    args = parser.parse_args()

    # ratio for testing set
    # default: 3
    args.test_set_ratio = 10 - args.train_set_ratio

    # show information
    print('{:{}}: {}'.format('Dataset', SPACE, args.dataset))
    print('{:{}}: {}'.format('Image directory name', SPACE, args.img_dir_name))
    print('{:{}}: {}'.format(
        'Training set ratio', SPACE, args.train_set_ratio))
    print('{:{}}: {}'.format('Testing set ratio', SPACE, args.test_set_ratio))

    # check valid or not
    if not os.path.isdir(args.dataset):
        parser.error('Wrong dataset path')
    if not os.path.isdir(args.dataset + '/' + args.img_dir_name):
        parser.error('Wrong image directory name')
    if args.train_set_ratio < 0 or args.train_set_ratio > 10:
        parser.error('Ration must between 0 and 10')

    return args


def make_pair_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # path to dataset
    # dafault: ../Dataset
    parser.add_argument('--dataset', type=str, default=basedir + '/Dataset',
                        help='path to dataset')
    # name for training set
    # default: train.txt
    parser.add_argument('--train_set_name', type=str, default='train.txt',
                        help='name for training set')
    # name for annotation directory
    # default: Annotations
    parser.add_argument('--ann_dir_name', type=str, default='Annotations',
                        help='name for annotation directory')
    # name for training pair
    # default: train_pairs.txt
    parser.add_argument('--train_pair_name', type=str,
                        default='train_pairs.txt',
                        help='name for training pair')

    args = parser.parse_args()

    # show information
    print('{:{}}: {}'.format('Dataset', SPACE, args.dataset))
    print('{:{}}: {}'.format('Training set name', SPACE, args.train_set_name))
    print('{:{}}: {}'.format('Annotation directory name',
                             SPACE, args.ann_dir_name))

    # check valid or not
    if not os.path.isdir(args.dataset):
        parser.error('Wrong dataset path')
    if not os.path.isfile(args.dataset + '/' + args.train_set_name):
        parser.error('Wrong training set name')
    if not os.path.isdir(args.dataset + '/' + args.ann_dir_name):
        parser.error('Wrong annotation name')

    return args


def grabcut_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # path to dataset
    # dafault: ../dataset
    parser.add_argument('--dataset', type=str, default=basedir + '/Dataset',
                        help='path to dataset')
    # path to image directory
    # default: ./dataset/JPEGimages
    parser.add_argument('--img_dir_name', type=str, default='JPEGImages',
                        help='name for image directory')
    # name for training pair
    # default: train_pairs.txt
    parser.add_argument('--train_pair_name', type=str,
                        default='train_pairs.txt',
                        help='name for training pair')
    # name for grabcut directory
    # default: inst_grabcut
    parser.add_argument('--grabcut_dir_name', type=str, default='Grabcut_inst',
                        help='name for grabcut directory')
    # name for img wiht grabcut directory
    # default: img_grabcuts
    parser.add_argument('--img_grabcuts_dir', type=str,
                        default='Grabcut_pairs',
                        help='name for image with grabcuts directory')
    # pool for multiprocess
    # default: 4
    parser.add_argument('--pool_size', type=int, default=4,
                        help='pool for multiprocess')
    # grabcut iteration
    # default: 3
    parser.add_argument('--grabcut_iter', type=int,
                        default=3, help='grabcut iteration')
    # label directory name
    # default: Segmentation_label
    parser.add_argument('--label_dir_name', type=str,
                        default='Segmentation_label',
                        help='name for label directory')
    args = parser.parse_args()

    # show information
    print('{:{}}: {}'.format('Dataset', SPACE, args.dataset))
    print('{:{}}: {}'.format('Image directory name', SPACE, args.img_dir_name))
    print('{:{}}: {}'.format('Pair name', SPACE, args.train_pair_name))
    print('{:{}}: {}'.format('Grabcut directory name', SPACE,
                             args.grabcut_dir_name))
    print('{:{}}: {}'.format('Label directory name', SPACE,
                             args.label_dir_name))

    # check valid or not
    if not os.path.isdir(args.dataset):
        parser.error('Wrong dataset path')
    if not os.path.isdir(args.dataset + '/' + args.img_grabcuts_dir):
        parser.error('Wrong image with grabcuts path')
    if not os.path.isdir(args.dataset + '/' + args.img_dir_name):
        parser.error('Wrong image directory name')
    if not os.path.isfile(args.dataset + '/' + args.train_pair_name):
        parser.error('Wrong pair name')
    if not os.path.isdir(args.dataset + '/' + args.grabcut_dir_name):
        parser.error('Wrong grabcut directory name')
    if not os.path.isdir(args.dataset + '/' + args.label_dir_name):
        parser.error('Wrong label directory name')

    return args


def model_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # path to dataset
    # dafault: ../dataset
    parser.add_argument('--dataset', type=str, default=basedir + '/Dataset',
                        help='path to dataset')
    # name for set
    # default: val.txt
    parser.add_argument('--set_name', type=str, default='val.txt',
                        help='name for set')
    # label directory name
    # default: Segmentation_label
    parser.add_argument('--label_dir_name', type=str,
                        default='Segmentation_label',
                        help='name for label directory')
    # path to image directory
    # default: ./dataset/JPEGimages
    parser.add_argument('--img_dir_name', type=str, default='JPEGImages',
                        help='name for image directory')
    # number of classes for segmentation
    # dafault: 21
    parser.add_argument('--classes', type=int, default=21,
                        help='number of classes for segmentation')
    # batch size for training
    # default: 16
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training')
    # epoch for training
    # dafault: 2000
    parser.add_argument('--epoch', type=int, default=2000,
                        help='epoch for training')
    # learning rate for training
    # default: 0.01
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training')
    # momentum for optimizer
    # default: 0.9
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    # probability for dropout
    # default: 0.5
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='probability for dropout')
    # training or testing
    # default: 0 means False
    parser.add_argument('--is_train', type=int, default=0,
                        help='training or testing [1 = True / 0 = False]')
    # step for saving weight
    # default: 2
    parser.add_argument('--save_step', type=int, default=2,
                        help='step for saving weight')
    # directory for prediction masks
    # default: ./dataset/Pred_masks
    parser.add_argument('--pred_dir_name', type=str, default='Pred_masks',
                        help='name for prediction masks directory')
    # directory for pairs
    # default: ./dataset/Pred_pairs
    parser.add_argument('--pair_dir_name', type=str, default='Pred_pairs',
                        help='name for pairs directory')
    # directory for crf prediction masks
    # default: ./dataset/CRF_masks
    parser.add_argument('--crf_dir_name', type=str, default='CRF_masks',
                        help='name for crf prediction masks directory')
    # directory for crf pairs
    # default: ./dataset/CRF_pred_pairs
    parser.add_argument('--crf_pair_dir_name', type=str, default='CRF_pairs',
                        help='name for crf pairs directory')
    # width for resize
    # default: 513
    parser.add_argument('--width', type=int, default=513,
                        help='width for resize')
    # height for resize
    # default: 513
    parser.add_argument('--height', type=int, default=513,
                        help='height for resize')
    # restore target
    # default: '0'
    parser.add_argument('--restore_target', type=str, default='0',
                        help='target for restore ')

    args = parser.parse_args()

    # show information
    print('{:{}}: {}'.format('Dataset', SPACE, args.dataset))
    print('{:{}}: {}'.format('Set name', SPACE, args.set_name))
    print('{:{}}: {}'.format('Label directory name',
                             SPACE, args.label_dir_name))
    print('{:{}}: {}'.format('Image directory name', SPACE, args.img_dir_name))
    print('{:{}}: {}'.format('Classes', SPACE, args.classes))
    print('{:{}}: {}'.format('Batch size', SPACE, args.batch_size))
    print('{:{}}: {}'.format('Epoch', SPACE, args.epoch))
    print('{:{}}: {}'.format('Learning rate', SPACE, args.learning_rate))
    print('{:{}}: {}'.format('Momentum', SPACE, args.momentum))
    print('{:{}}: {}'.format('Probability', SPACE, args.keep_prob))
    print('{:{}}: {}'.format('Training',
                             SPACE, True if args.is_train else False))
    print('{:{}}: {}'.format('Save step', SPACE, args.save_step))
    print('{:{}}: {}'.format('Prediction masks directory name', SPACE,
                             args.pred_dir_name))
    print('{:{}}: {}'.format('Prediction pairs directory name', SPACE,
                             args.pair_dir_name))
    print('{:{}}: {}'.format('CRF prediction masks directory name', SPACE,
                             args.crf_dir_name))
    print('{:{}}: {}'.format('CRF prediction pairs directory name', SPACE,
                             args.crf_pair_dir_name))
    print('{:{}}: {}'.format('Width for resize', SPACE, args.width))
    print('{:{}}: {}'.format('Height for resize', SPACE, args.height))
    print('{:{}}: {}'.format('Target for restore', SPACE, args.restore_target))

    # check valid or not
    if not os.path.isdir(args.dataset):
        parser.error('Wrong dataset path')
    if not os.path.isfile(args.dataset + '/' + args.set_name):
        parser.error('Wrong set name')
    if not os.path.isdir(args.dataset + '/' + args.label_dir_name):
        parser.error('Wrong label directory name')
    if not os.path.isdir(args.dataset + '/' + args.img_dir_name):
        parser.error('Wrong image directory name')
    if not os.path.isdir(args.dataset + '/' + args.pred_dir_name):
        parser.error('Wrong prediction directory name')
    if not os.path.isdir(args.dataset + '/' + args.pair_dir_name):
        parser.error('Wrong prediction pairs directory name')
    if not os.path.isdir(args.dataset + '/' + args.crf_dir_name):
        parser.error('Wrong crf prediction directory name')
    if not os.path.isdir(args.dataset + '/' + args.crf_pair_dir_name):
        parser.error('Wrong crf prediction pairs directory name')

    return args


def mIoU_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # path to dataset
    # dafault: ../dataset
    parser.add_argument('--dataset', type=str, default=basedir + '/Dataset',
                        help='path to dataset')
    # name for set
    # default: val.txt
    parser.add_argument('--set_name', type=str, default='val.txt',
                        help='name for set')
    # name of ground truth directory
    # default: SegmentationClass
    parser.add_argument('--GT_dir_name', type=str, default='SegmentationClass',
                        help='name for ground truth directory')
    # name of prediction directory
    # default: CRF_masks
    parser.add_argument('--Pred_dir_name', type=str, default='CRF_masks',
                        help='name for prediction directory')
    # number of classes
    # default: 21
    parser.add_argument('--classes', type=int, default=21,
                        help='number of classes')

    args = parser.parse_args()

    # show information
    print('{:{}}: {}'.format('Dataset', SPACE, args.dataset))
    print('{:{}}: {}'.format('Set name', SPACE, args.set_name))
    print('{:{}}: {}'.format('ground truth directory name', SPACE,
                             args.GT_dir_name))
    print('{:{}}: {}'.format('Prediction directory name', SPACE,
                             args.Pred_dir_name))
    print('{:{}}: {}'.format('Classes', SPACE, args.classes))

    # check valid or not
    if not os.path.isdir(args.dataset):
        parser.error('Wrong dataset path')
    if not os.path.isfile(args.dataset + '/' + args.set_name):
        parser.error('Wrong set name')
    if not os.path.isdir(args.dataset + '/' + args.GT_dir_name):
        parser.error('Wrong ground truth directory name')
    if not os.path.isdir(args.dataset + '/' + args.Pred_dir_name):
        parser.error('Wrong prediction directory name')
    return args


def boxi_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # path to dataset
    # dafault: ../dataset
    parser.add_argument('--dataset', type=str, default=basedir + '/Dataset',
                        help='path to dataset')
    # path to annotations directory
    # default: ./dataset/Annotations
    parser.add_argument('--ann_dir_name', type=str, default='Annotations',
                        help='name for annotations directory')
    # name for set name
    # default: train.txt
    parser.add_argument('--set_name', type=str,
                        default='train.txt',
                        help='name for set')
    # label directory name
    # default: Segmentation_label
    parser.add_argument('--label_dir_name', type=str,
                        default='Segmentation_label',
                        help='name for label directory')
    args = parser.parse_args()

    # show information
    print('{:{}}: {}'.format('Dataset', SPACE, args.dataset))
    print('{:{}}: {}'.format('Annotations directory name', SPACE,
                             args.ann_dir_name))
    print('{:{}}: {}'.format('Set name', SPACE, args.set_name))
    print('{:{}}: {}'.format('Label directory name', SPACE,
                             args.label_dir_name))

    # check valid or not
    if not os.path.isdir(args.dataset):
        parser.error('Wrong dataset path')
    if not os.path.isdir(args.dataset + '/' + args.ann_dir_name):
        parser.error('Wrong annotations directory name')
    if not os.path.isfile(args.dataset + '/' + args.set_name):
        parser.error('Wrong set name')
    if not os.path.isdir(args.dataset + '/' + args.label_dir_name):
        parser.error('Wrong label directory name')

    return args
