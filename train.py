from src.src_main import SPNet
from src.utils import DDPutils
from config import Configs
import os
import torch
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # gpus

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(
    "options for SPNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model_type",
    type=str,
    default="Large",
    help="Model Version: [Tiny, Small, Base, Large]",
)
args = parser.parse_args()
MODEL_TYPE = args.model_type


def DDP_main(rank, world_size):
    cf = Configs(world_size, model_type=MODEL_TYPE)

    # DDP components
    DDPutils.setup(rank, world_size, 6003)
    if rank == 0:
        print(f"Selected arguments: {cf.__dict__}")

    trainer = SPNet(cf, rank=rank)
    trainer.train(cf)
    DDPutils.cleanup()


if __name__ == "__main__":
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        DDPutils.run_demo(DDP_main, n_gpus)
