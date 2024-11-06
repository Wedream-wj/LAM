# --------------------------------------------------------------------------------
# Analyze super-resolution models using LAM.
# Official GitHub: https://github.com/X-Lowlevel-Vision/LAM_Demo
#
# Modified by Jinpeng Shi (https://github.com/jinpeng-s)
# --------------------------------------------------------------------------------
# Modified by FriedRiceLab (https://github.com/Fried-Rice-Lab/FriedRiceLab)
# Paper: <<Interpreting Super-Resolution Networks with Local Attribution Maps>>
# --------------------------------------------------------------------------------
# python interpret.py --img_path "demo/Urban7/7.png"
# --------------------------------------------------------------------------------
import os.path
from os import path as osp

# from utils import parse_options, make_exp_dirs, get_model_interpretation
from utils import get_model_interpretation
from model import *
import argparse

parser = argparse.ArgumentParser(description="Draw local attribution maps(LAM)")

parser.add_argument("--img_path", type=str, default="demo/Urban7/7.png", help="归因HR图像路径")
parser.add_argument("--patch_x", type=int, default=110, help="patch的x坐标")
parser.add_argument("--patch_y", type=int, default=150, help="patch的y坐标")
parser.add_argument("--window_size", type=int, default=16, help="patch的尺寸")
parser.add_argument("--output_dir", type=str, default="./output", help="输出文件夹")
# parser.add_argument("--checkpoint", type=str, default=None)

args = parser.parse_args()

def interpret_pipeline(root_path):  # noqa

    # create model
    # teacher_1 = get_model(model_name="IMDN", upscale=4, checkpoint="IMDN_x4.pth").to(
    #     "cuda")
    model1 = get_model(model_name="EdgeSRN", upscale=4, checkpoint="EdgeSRN_x4.pth").to(
        "cuda")

    # call interpretation method to get LAM
    # img:LAM, di:diffusion index
    img, di = get_model_interpretation(model=model1, img_path=args.img_path,
                                       w=args.patch_x, h=args.patch_y,
                                       use_cuda=True)

    # create output_dir
    os.makedirs(osp.join(args.output_dir), exist_ok=True)
    # path to save LAM
    save_path = osp.join(args.output_dir, args.img_path.split("/")[-1].replace('.png','_interpret.png'))
    # save LAM
    img.save(save_path)
    # a larger DI indicates more pixels
    print("DI : {}".format(di))
    print("The LAM result are saved to {}".format(str(save_path)))
    print("Successfully!")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    interpret_pipeline(root_path)
