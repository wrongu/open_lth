# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from foundations import hparams
from lottery.desc import LotteryDesc
from pruning import sparse_global
from models.cifar_resnet import Model as CIFARModel


class Model(CIFARModel):
    """A residual neural network as originally designed for Tiny ImageNet(Modified from CIFAR-10)."""
    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__(plan, initializer, outputs=None)
        self.relu = nn.ReLU()
        outputs = outputs or 200

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Model.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    @staticmethod
    def hidden_numel_from_model_name(model_name):
        plan = Model.plan_from_model_name(model_name)
        ch, h, w = 3, 64, 64
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
        numel += 200
        return numel

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('tinyimagenet_resnet_') and
                5 > len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]) and
                (int(model_name.split('_')[2]) - 2) % 6 == 0 and
                int(model_name.split('_')[2]) > 2)

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=None):
        """The naming scheme for a ResNet is 'tinyimagenet_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'tinyimagenet_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'tinyimagenet_resnet_20' or 'tinyimagenet_resnet_20_16'.
        """
        plan = Model.plan_from_model_name(model_name)
        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='tinyimagenet_resnet_20',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='tinyimagenet',
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
