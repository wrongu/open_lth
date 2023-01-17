# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy


def maybe_get_arg(arg_name, positional=False, position=0, boolean_arg=False):
    parser = argparse.ArgumentParser(add_help=False)
    prefix = '' if positional else '--'
    if positional:
        for i in range(position):
            parser.add_argument(f'arg{i}')
    if boolean_arg:
        parser.add_argument(prefix + arg_name, action='store_true')
    else:
        parser.add_argument(prefix + arg_name, type=str, default=None)
    try:
        args = parser.parse_known_args()[0]
        return getattr(args, arg_name) if arg_name in args else None
    except:
        return None


def fix_all_seed(args):
    """Fix all seed and modify the seed parameters to be consistent
    Args:
        args (argparse.Namespace): The arguments.
    """
    import random
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    args = copy.deepcopy(args)
    seed = getattr(args, 'seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        args.data_order_seed = seed
        args.transformation_seed = seed
    return args

    

