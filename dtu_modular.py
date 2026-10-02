import os
import numpy as np
import argparse
from src.utils import read_pair, read_images, read_projection_txt
from src.cv import spriteMVS

basedir = '/home/asclab/projects/MVS/datasets/dtu/dtu/'
calib_data = "/home/asclab/projects/MVS/datasets/dtu/cal18"
pairfile = "/home/asclab/projects/MVS/datasets/dtu/mvs_training/dtu/Cameras/pair.txt"

redundancy = 5
threshold = 0.55

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=basedir)
    parser.add_argument('--calib_data', type=str, default=calib_data)
    parser.add_argument('--pairfile', type=str, default=pairfile)
    parser.add_argument('--redundancy', type=int, default=redundancy)
    parser.add_argument('--threshold', type=float, default=threshold)
    parser.add_argument('--octreeN', type=int, default= 3)
    parser.add_argument('--downsampleFactor', type=int, default= 5)
    parser.add_argument('--imageIdx', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()

    fn = sorted(os.listdir(args.basedir))[0]
    img_path = args.basedir + fn + '/images/'
    n = int(fn[4:])

    pairs = read_pair(args.pairfile).astype(int)
    total_img = read_images(img_path)
    total_p = read_projection_txt(args.calib_data)[1]

    img_src = total_img[args.imageIdx]
    p_src = total_p[args.imageIdx]
    img_ref = [total_img[idx] for idx in pairs[args.imageIdx]]
    p_ref = [total_p[idx] for idx in pairs[args.imageIdx]]

    img_ref.insert(5, img_src)
    p_ref.insert(5, p_src)

    mvs = spriteMVS(
        dmin=300,
        dmax=1000,
        convSize=5,
        nPlane=20,
        factor=args.downsampleFactor,
        row=total_img[0].shape[0],
        col=total_img[0].shape[1],
        thresh=args.threshold,
        rdd=args.redundancy
    )

    for idxx, img_iter in enumerate(img_ref):
        mvs.register(img_iter, p_ref[idxx])

    depth_idx = mvs.sparse()

    mvs.nPlane = 10

    dd, depth_idx = mvs.octree(args.octreeN)

    os.makedirs(f'./result/result_{args.octreeN}', exist_ok=True)
    output_file = f'./result/result_{args.octreeN}/{args.imageIdx}.ply'
    mvs.save_depth_ply(output_file)


if __name__ == "__main__":
    main()