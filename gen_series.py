import os
import numpy as np
import argparse
from src.utils import read_pair, read_images, read_projection_txt
from src.cv import spriteMVS


'''
Use this script if you want to generate point cloud from a series of images.
Series of images have to be geometrically good pairs, for example, if you are taking a video
of a target, then the consecutive frames should have sufficient overlap and good geometric correspondence.

'''

redundancy = 5
threshold = 0.5
basedir = '/home/asclab/projects/MVS/datasets/dtu/dtu/scan1/images'
projectiondir = './assets/total_p_dtu.npy'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgFolder', type=str, default=basedir)
    parser.add_argument('--projectionFile', type=str, default=projectiondir)
    parser.add_argument('--redundancy', type=int, default=redundancy)
    parser.add_argument('--threshold', type=float, default=threshold)
    parser.add_argument('--octreeN', type=int, default= 3)
    parser.add_argument('--downsampleFactor', type=int, default= 1)
    return parser.parse_args()



def main():
    args = parse_args()

    total_img = read_images(args.imgFolder)
    total_p = np.load(args.projectionFile)

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

    imgN = 0

    for img_src, p_src in zip(total_img, total_p):
        mvs.register(img_src, p_src)

        if len(mvs.imglist) > 10: # when number of images equal to 11 (10 ref, 1 src; at index 5)
            mvs.nPlane = 20
            depth_idx = mvs.sparse()
            mvs.nPlane = 10
            dd, depth_idx = mvs.octree(args.octreeN)

            os.makedirs(f'./result', exist_ok=True)
            output_file = f'./result/{imgN:04d}.ply'
            mvs.save_depth_ply(output_file)
            imgN += 1

            mvs.unregister()

if __name__ == "__main__":
    main()