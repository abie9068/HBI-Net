# Copyright (c) HBI-Net authors. All rights reserved.
"""Evaluation entry for HBI-Net.

Thin wrapper around Open-CD's test tool. Imports ``hbi_net`` to register the
custom modules, then runs evaluation with the standard MMEngine ``Runner``.

Usage::

    python tools/test.py configs/hbi_net_abcd.py work_dirs/hbi_net/best.pth
"""
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# Register all HBI-Net custom modules into the Open-CD registry.
import hbi_net  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description='Test HBI-Net')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--show-dir', help='directory to save visualized predictions')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.load_from = args.checkpoint
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.show_dir is not None:
        cfg.default_hooks.visualization.draw = True
        cfg.default_hooks.visualization.show = False
        cfg.visualizer.save_dir = args.show_dir

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
