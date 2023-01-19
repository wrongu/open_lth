# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

from platforms import base


class Platform(base.Platform):
    @property
    def root(self):
        return '/data/open_lth/'

    @property
    def dataset_root(self):
        return '/data/'

    @property
    def imagenet_root(self):
        return '/data/imagenet/'
