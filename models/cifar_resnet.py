# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """A residual neural network as originally designed for CIFAR-10."""

    # The following properties can be overridden by subclasses to change the behavior of the class
    KERNEL_SIZE = 3
    MODEL_NAME = "cifar_resnet"  # subclasses need to be more specific
    DATASET = "cifar10"
    CLASSES = 10
    INPUT_CH = 3
    INPUT_H = 32
    INPUT_W = 32

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False, kernel_size=3):
            super(Model.Block, self).__init__()
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=kernel_size, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return self.relu2(out)

    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        outputs = outputs or self.CLASSES

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=self.KERNEL_SIZE, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Model.Block(current_filters, filters, downsample, kernel_size=self.KERNEL_SIZE))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @classmethod
    def is_valid_model_name(cls, model_name):
        base_name_len = len(cls.MODEL_NAME.split('_'))
        return (model_name.startswith(cls.MODEL_NAME) and
                base_name_len+3 > len(model_name.split('_')) > base_name_len and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[base_name_len:]]) and
                (int(model_name.split('_')[base_name_len]) - 2) % 6 == 0 and
                int(model_name.split('_')[base_name_len]) > 2)

    @classmethod
    def hidden_numel_from_model_name(cls, model_name):
        plan = cls.plan_from_model_name(model_name)
        ch, h, w = cls.INPUT_CH, cls.INPUT_H, cls.INPUT_W
        numel = ch * h * w
        # Initial conv/bn/relu
        ch = plan[0][0]
        numel += ch * h * w * 3
        # Blocks
        for segment_index, (new_ch, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                has_projection = downsample or new_ch != ch
                ch = new_ch
                if downsample:
                    h, w = h/2, w/2
                # Each block is 7 layers, plus maybe two projection stages: conv/bn/relu/conv/bn/(proj?)/skip/relu
                numel += ch * h * w * (9 if has_projection else 7)
        # FC
        numel += cls.CLASSES
        return numel

    @classmethod
    def plan_from_model_name(cls, model_name):
        if not cls.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        base_name_len = len(cls.MODEL_NAME.split('_'))
        W = 16 if len(name) == base_name_len + 1 else int(name[base_name_len+1])
        D = int(name[base_name_len])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]
        return plan

    @classmethod
    def get_model_from_name(cls, model_name, initializer,  outputs=None):
        """If MODEL_NAME is 'cifar_resnet', then the naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """
        plan = cls.plan_from_model_name(model_name)
        return cls(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @classmethod
    def default_hparams(cls):
        model_hparams = hparams.ModelHparams(
            model_name=f'{cls.MODEL_NAME}_20',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name=cls.DATASET,
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
