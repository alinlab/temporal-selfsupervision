import os
import argparse
import sys

from test import test
from train import train
from configs.defaults import get_cfg

import research_platform.utils.checkpoint as cu
from research_platform.utils.misc import launch_job

""""
General launcher script for Emprical Risk Minimization (i.e., supervised learning) cases.
"""

def parse_args(default=False):
    parser = argparse.ArgumentParser(
        description='Parse arguments')
    parser.add_argument(
        "--gpu_ids",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        type=str,
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See configs/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    
    return parser.parse_args()

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    output_dir = cfg.OUTPUT_DIR
    for name in cfg.DATA.APPEND_TO_OUTPUT_DIRNAME:
        output_dir += f'_{name}-{cfg.DATA[name]}'
    cfg.OUTPUT_DIR = output_dir
    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def main():
    """ argument define """
    args = parse_args()

    """ set torch device"""
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES" ]= args.gpu_ids
    
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    if cfg.ERM_TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    if cfg.ERM_TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()
