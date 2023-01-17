# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

from platforms import base


class Platform(base.Platform):
    @property
    def root(self):
        _root = '/project/representation-paths/checkpoints'
        if getattr(self, 'seed', None) is not None:
            return os.path.join(_root, f'seed_{self.seed}')
        else:
            return _root

    @property
    def dataset_root(self):
        return '/project/representation-paths/data'

    @property
    def imagenet_root(self):
        return '/project/representation-paths/data/imagenet/'

    def seed_save_path(self, seed):
        """Add seed as suffix to checkpoint path."""
        self.seed = seed
