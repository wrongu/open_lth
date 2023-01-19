from models.cifar_resnet import Model as CifarResnet


class Model(CifarResnet):
    """A residual neural network as originally designed for Tiny ImageNet(Modified from CIFAR-10)."""

    # All we need to do to adapt the cifar resnet to tinyimagenet is change some of the class-level constants.
    KERNEL_SIZE = 3
    MODEL_NAME = "tinyimagenet_resnet"
    DATASET = "tinyimagenet"
    CLASSES = 200
    INPUT_CH = 3
    INPUT_H = 64
    INPUT_W = 64
