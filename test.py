import os
import argparse

from PIL import Image
from pathlib import Path
from src.networks import V2Net
from test_utils import DataReader, metrics
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpus

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def model_config(model_type):
    if model_type == "Tiny":
        dims = [96, 192, 384, 768]  # dimensions
        depths = [3, 3, 9, 3]  # block number
        dp_rate = 0.0
    if model_type == "Small":
        dims = [96, 192, 384, 768]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.1
    elif model_type == "Base":
        dims = [128, 256, 512, 1024]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.1
    elif model_type == "Large":
        dims = [192, 384, 768, 1536]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.2
    return dims, depths, dp_rate


def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for SPNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rgbd_dir",
        type=lambda x: Path(x),
        default="Test_Datasets/Ibims",
        help="Path to RGBD folder",
    )
    parser.add_argument(
        "--model_dir",
        type=lambda x: Path(x),
        default="checkpoints/models/Large_300.pth",
        help="Path to load models",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Large",
        help="Model Version: [Tiny, Small, Base, Large]",
    )
    parser.add_argument(
        "--norm_type",
        default="CNX",
        help="adopting SP-Normalization",
    )
    parser.add_argument(
        "--save_results",
        type=bool,
        default=True,
        help="save results",
    )

    args = parser.parse_args()
    return args


@torch.no_grad()
def demo(args):
    print("-----------building model-------------")
    dims, depths, dp_rate = model_config(args.model_type)
    network = V2Net(dims, depths, dp_rate, args.norm_type).cuda().eval()
    network.load_state_dict(torch.load(args.model_dir)["network"])
    raw_dirs = ["10%", "1%", "0.1%"]
    avg_rmse, avg_rel = 0.0, 0.0
    print("-----------inferring---------------")
    for raw_dir in raw_dirs:
        # init
        count, rmse, rel = 0.0, 0.0, 0.0

        for file in (args.rgbd_dir / "rgb").rglob("*.png"):
            str_file = str(file)
            raw_path = str_file.replace("/rgb/", "/raw_" + raw_dir + "/")
            save_path = str_file.replace("/rgb/", "/result_" + raw_dir + "/")
            gt_path = str_file.replace("/rgb/", "/gt/")
            data_reader = DataReader("cuda")
            # processing
            rgb, raw, gt, hole_raw = data_reader.read_data(str_file, raw_path, gt_path)
            pred = network(rgb, raw, hole_raw)

            # metrics
            count += 1.0
            rmse_temp, rel_temp = metrics(pred, gt)
            rmse += rmse_temp
            rel += rel_temp

            # save img
            if args.save_results:
                pred = data_reader.toint32(pred)
                os.makedirs(str(Path(save_path).parent), exist_ok=True)
                Image.fromarray(pred).save(save_path)
                print(raw_path)
        # metric each raw_dir
        rmse /= count
        rel /= count
        avg_rmse += rmse
        avg_rel += rel
    # average metric
    avg_rmse /= len(raw_dirs)
    avg_rel /= len(raw_dirs)
    print("Average: RMSE=", str(avg_rmse), " Rel=", str(avg_rel))


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
