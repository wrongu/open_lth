import os
import numpy as np
import concurrent
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform

def download_dataset(root):
    """get tiny imagenet dataset from huggingface given by split"""
    # check if the dataset is already downloaded
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
        # Retrieve data directly from Stanford data source
        os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P ' + root)
        os.system(f'cd {root} && unzip -qq "tiny-imagenet-200.zip"')

        # rearrange the validation label
        val_img_dir = os.path.join(root, 'tiny-imagenet-200', 'val', 'images')

        # Open and read val annotations text file
        with open(os.path.join(os.path.join(root, 'tiny-imagenet-200', 'val'), 'val_annotations.txt'), 'r') as f:
            data = f.readlines()
            # Create dictionary to store img filename (word 0) and corresponding
            # label (word 1) for every line in the txt file (as key value pair)
            val_img_dict = {}
            for line in data:
                words = line.split('\t')
                val_img_dict[words[0]] = words[1]
        # Create subfolders (if not present) for validation images based on label,
        # and move images into the respective folders
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(val_img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

class Dataset(base.ImageDataset):
    """The Tiny Imagenet dataset."""

    @staticmethod
    def num_train_examples(): return 100000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 200

    @staticmethod
    def get_train_set(use_augmentation):
        transforms = Dataset._augment_transforms() if use_augmentation else Dataset._transforms()
        download_dataset(root=os.path.join(get_platform().dataset_root, 'tinyimagenet'))
        return Dataset(os.path.join(get_platform().dataset_root, 'tinyimagenet', 'tiny-imagenet-200', 'train'),
                       transforms)

    @staticmethod
    def get_test_set():
        download_dataset(root=os.path.join(get_platform().dataset_root, 'tinyimagenet'))
        return Dataset(os.path.join(get_platform().dataset_root, 'tinyimagenet', 'tiny-imagenet-200', 'val', 'images'),
                       Dataset._transforms())

    @staticmethod
    def _augment_transforms():
        return [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(64, 8)]

    @staticmethod
    def _transforms():
        return []

    def __init__(self, loc, image_transforms):
        # Load the data.
        dataset = torchvision.datasets.ImageFolder(loc, transform=image_transforms)

        examples, labels = zip(*dataset.samples)
        super(Dataset, self).__init__(
            np.array(examples), np.array(labels), image_transforms,
            [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @staticmethod
    def example_to_image(example):
        with get_platform().open(example, 'rb') as fp:
            return Image.open(fp).convert('RGB')


DataLoader = base.DataLoader
