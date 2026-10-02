import cupy as cp
import numpy as np
import time
import cv2
from cupyx.scipy.ndimage import sobel, median_filter
from plyfile import PlyData, PlyElement
from .utils import get_hom, get_CV_matrices, fill_nan_nearest

class spriteMVS:
    def __init__(self, dmin, dmax, convSize, nPlane, factor, row, col, thresh, rdd):
        self.convSize, self.nPlane = convSize, nPlane
        self.factor = factor
        self.row, self.col = int(row/factor), int(col/factor)
        self.thresh = thresh
        self.rdd = rdd
        self.dmin = dmin
        self.nPlane = nPlane
        self.dmax = dmax

        
        
        self.zList = []
        self.plist = []
        self.klist = []
        self.tlist = []
        self.rlist = []
        self.imglist = []
        self.gradlist = []
        self.imgflatlist = []
        self.gradflatlist = []
        self.planesInit = []

        self.argset()

    def argset(self):
        # Rmesh, Cmesh = np.meshgrid(np.arange(self.row), np.arange(self.col))
        Rmesh, Cmesh = np.meshgrid(np.arange(self.row), np.arange(self.col), indexing= 'ij')
        self.cr1 = cp.asarray(get_hom(np.column_stack((Cmesh.flatten(), Rmesh.flatten())))).T

    def register(self, img, p):
        
        
        kernelSize = self.convSize

        k, r, t, _ = get_CV_matrices(p)
        
        
        # Downsample and get Cupy list for both RGB and Grad
        imgDown = cp.asarray(downsample_rgb(img, factor = self.factor, kernel_size = kernelSize), dtype = cp.float32)
        gradDown = cp.asarray(get_grad_world_gpu(imgDown, r), dtype = cp.float32)

        imgDown /= 255.0
        gradDown /= 255.0
        
        # Scale intrinsics based on downsample    
        newK = scale_intrinsics(k, self.factor)
        newP = newK @ np.column_stack((r,t))

        self.tlist.append(cp.array(t))
        self.rlist.append(cp.array(r))
        self.plist.append(cp.array(newP))
        self.klist.append(cp.array(newK))
        self.imglist.append(cp.array(imgDown))
        self.gradlist.append(cp.array(gradDown))
        

        cr1 = self.cr1
        pixY = cr1[1].astype(cp.int32)
        pixX = cr1[0].astype(cp.int32)

        rgbFlat = imgDown[pixY, pixX]
        gradFlat = gradDown[pixY, pixX]

        self.imgflatlist.append(cp.array(rgbFlat))
        self.gradflatlist.append(cp.array(gradFlat))

    def unregister(self):
        
        self.rlist = self.rlist[1:]
        self.plist = self.plist[1:]
        self.klist = self.klist[1:]
        self.tlist = self.tlist[1:]
        self.imglist = self.imglist[1:]
        self.gradlist = self.gradlist[1:]
        self.imgflatlist = self.imgflatlist[1:]
        self.gradflatlist = self.gradflatlist[1:]


    def getPlanes(self):
        kInv = cp.linalg.inv(self.klist[5])
        ray = kInv @ self.cr1

        z = self.dList[:, None, None]
        
        pts = self.rlist[5].T @ (z * ray - self.tlist[5])
        ones = cp.ones((self.nPlane, 1, pts.shape[-1]))
        pts = cp.concatenate([pts,ones], axis = 1)
        return pts

    def getPlanesOct(self):
        kInv = cp.linalg.inv(self.klist[5])
        t_src = self.tlist[5]
        r_src = self.rlist[5]
        ray = kInv @ self.cr1
        
        pts = r_src.T[None, :, :] @ (self.ll[:, None, :]* ray[None, :, :] - t_src[None, :, :])
        ones = cp.ones((self.nPlane, 1, pts.shape[-1]))
        pts = cp.concatenate([pts,ones], axis = 1)
        return pts
            
        
    def sparse(self):

        self.dList = cp.linspace(self.dmin, self.dmax, self.nPlane)
        self.dd = self.dList[1] - self.dList[0]


        pts = self.getPlanes()

        startTime = time.time()

        rgbPair, gradPair = getRgbPair(
            pts, 
            self.plist, 
            self.imglist, 
            self.gradlist
            )

        rgbSim = similarityRGB(rgbPair, self.imgflatlist[5][None, None, :, :])
        gradSim = similarityGrad(gradPair, self.gradflatlist[5][None, None, :, :])
        combinedSim = gradSim * rgbSim
        # combinedSim = gradSim
        
        combinedSim = combinedSim.reshape(combinedSim.shape[0], combinedSim.shape[1], self.row, self.col)

        combinedSim = median_filter(combinedSim, size = (1, 1, 5, 5)).reshape(combinedSim.shape[0], -1, self.col * self.row)        # (ncam, nplane, ncol*nrow)
        
        depthIdx = self.score_oct(combinedSim)
        # print(combinedSim.shape)
        # depthIdx = cp.argmax(cp.nanmean(combinedSim, axis = 0), axis = 0)
        
        endTime = time.time()
        print(f'Sparse: {(endTime-startTime):04f} seconds')
        
        self.depthIdx = depthIdx
        
        return depthIdx


    def octree(self, N):
        idx = self.depthIdx.reshape(self.row, self.col)

        maskSparse = idx == -1

        result = self.dList[idx].copy()
        result[maskSparse] = cp.nan

        # plt.imshow(result.get().reshape(self.row, self.col))
        # plt.colorbar()
        # plt.show()

        self.ddList = cp.full(
            (self.row, self.col),
            self.dd,
            dtype=cp.float32
        )

        for iteration in range(N):
            startTime = time.time()

            # Use neighboring valid depth only as the search center
            l = fill_nan_nearest(
                result,
                cp.isnan(result)
            )
            l = median_filter(l, size = 5)

            lower = l - self.ddList
            upper = l + self.ddList

            if iteration == 0:
                lower[maskSparse] = self.dList[0] + self.dd
                upper[maskSparse] = self.dList[-1] - self.dd

            self.ll = cp.linspace(
                lower,
                upper,
                self.nPlane,
                axis=0
            ).reshape(self.nPlane, -1)

            pts = self.getPlanesOct()

            rgbPair, gradPair = getRgbPair(
                pts,
                self.plist,
                self.imglist,
                self.gradlist
            )

            rgbSim = similarityRGB(
                rgbPair,
                self.imgflatlist[5][None, None, :, :]
            )

            gradSim = similarityGrad(
                gradPair,
                self.gradflatlist[5][None, None, :, :]
            )

            combinedSim = rgbSim * gradSim

            combinedSim = combinedSim.reshape(
                combinedSim.shape[0],
                combinedSim.shape[1],
                self.row,
                self.col
            )

            combinedSim = median_filter(
                combinedSim,
                size=(1, 1, 5, 5)
            )

            combinedSim = combinedSim.reshape(
                combinedSim.shape[0],
                combinedSim.shape[1],
                self.row * self.col
            )

            depthIdx = self.score_oct(combinedSim)

            # IMPORTANT: determine validity BEFORE indexing
            good_now = (
                depthIdx != -1
            ).reshape(self.row, self.col)

            # Never use -1 directly for indexing
            safe_depth_idx = cp.clip(
                depthIdx,
                0,
                self.nPlane - 1
            )

            result_new = cp.take_along_axis(
                self.ll.reshape(
                    1,
                    self.nPlane,
                    self.row,
                    self.col
                ),
                safe_depth_idx.reshape(
                    1,
                    1,
                    self.row,
                    self.col
                ),
                axis=1
            )[0, 0]

            # Only accept newly valid estimates
            result[good_now] = result_new[good_now]

            # Keep previous depth for failed pixels

            # Narrow successful searches
            self.ddList[good_now] *= 0.8

            # # Narrow failed searches more slowly
            self.ddList[~good_now] *= 0.95

            # if self.nPlane > 1:
            #     self.nPlane -= 1

            # self.nPlane 

            self.ddList = cp.maximum(
                self.ddList,
                1e-4
            )

            cp.cuda.Stream.null.synchronize()

            endTime = time.time()

            print(
                f"Octree iteration {iteration + 1}/{N}: "
                f"{endTime - startTime:.4f} s"
            )

            # plt.imshow(result.get().reshape(self.row, self.col))
            # plt.colorbar()
            # plt.show()

        self.depth = result

        return self.ll, result


    def score(self, simm):

        simm = cp.nan_to_num(simm, nan=0.0)
        simm = cp.where(simm < self.thresh, 0, simm)
        pred_classes = cp.argmax(simm, axis=1)

        C = simm.shape[1]
        one_hot = pred_classes[:, None, :] == cp.arange(C)[None, :, None]
        votes = cp.sum(one_hot, axis=0)
        votes = cp.where(votes < self.rdd, 0, votes)

        return votes

    def score_oct(self, simm):

        simm = cp.nan_to_num(simm, nan=0.0)
        simm = cp.where(simm < self.thresh, 0, simm)
        pred_classes = cp.argmax(simm, axis=1)

        C = simm.shape[1]
        one_hot = pred_classes[:, None, :] == cp.arange(C)[None, :, None]
        votes = cp.sum(one_hot, axis=0)
        votes = cp.where(votes < self.rdd, 0, votes)
        majority = cp.argmax(votes, axis=0)
        majority[majority == 0] = -1
        
        return majority

    def save_depth_ply(self, filename="depth.ply"):
        depth = cp.asnumpy(self.depth)
        K = cp.asnumpy(self.klist[5])
        R = cp.asnumpy(self.rlist[5])
        t = cp.asnumpy(self.tlist[5]).reshape(3)
        img = cp.asnumpy(self.imglist[5])

        H, W = depth.shape
        yy, xx = np.meshgrid(
            np.arange(H),
            np.arange(W),
            indexing='ij'
        )

        valid = np.isfinite(depth) & (depth > 0)

        if np.count_nonzero(valid) == 0:
            print("No valid depth points found.")
            return

        z = depth[valid]
        u = xx[valid]
        v = yy[valid]

        # Back-project pixels into camera coordinates
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]

        pts_cam = np.column_stack((x, y, z)).astype(np.float32)

        # Camera -> world
        # X_cam = R X_world + t
        # X_world = R.T (X_cam - t)
        pts_world = (pts_cam - t[None, :]) @ R

        # RGB
        colors = img[valid]

        if colors.max() <= 1.0:
            colors = colors * 255.0

        colors = np.clip(colors, 0, 255).astype(np.uint8)

        # PLY vertex structure
        vertex = np.empty(
            len(pts_world),
            dtype=[
                ('x', 'f4'),
                ('y', 'f4'),
                ('z', 'f4'),
                ('red', 'u1'),
                ('green', 'u1'),
                ('blue', 'u1')
            ]
        )

        vertex['x'] = pts_world[:, 0]
        vertex['y'] = pts_world[:, 1]
        vertex['z'] = pts_world[:, 2]
        vertex['red'] = colors[:, 0]
        vertex['green'] = colors[:, 1]
        vertex['blue'] = colors[:, 2]

        

        PlyData([
            PlyElement.describe(vertex, 'vertex')
        ], text=True).write(filename)

        print(f"Saved {len(pts_world)} points to {filename}")
    
def getRgbPair(sparseVoxel, plistGpu, imgDownsample, gradDownsample):
    
    row, col, _ = imgDownsample[0].shape
        
    plistIter = cp.asarray(plistGpu[:5] + plistGpu[6:])
    imgIter = cp.asarray(imgDownsample[:5] + imgDownsample[6:])
    gradIter = cp.asarray(gradDownsample[:5] + gradDownsample[6:])

    projIter = plistIter[:, None, :, :]@sparseVoxel[None, :, :]
    projIter = cp.rint((projIter / projIter[:,:,-1,:][:,:,None,:])[:,:,:-1,:]).astype(cp.int32)
    cIter, rIter = projIter[:,:,0,:], projIter[:,:,1,:]
    invalid = (cIter < 0) | (cIter >= col) | (rIter < 0) | (rIter >= row)
    cIter = cp.clip(cIter, 0, col-1)
    rIter = cp.clip(rIter, 0, row-1)  

    bIdx = cp.arange(imgIter.shape[0])[:, None, None]
    rgbSelect = imgIter[bIdx,rIter, cIter].astype(cp.float16)
    gradSelect = gradIter[bIdx,rIter, cIter].astype(cp.float16)

    rgbSelect[invalid] = cp.nan
    gradSelect[invalid] = cp.nan
    
    return rgbSelect, gradSelect

def downsample_rgb(img, factor=2, kernel_size=3):
    """
    Reduce the resolution of an RGB image using convolution (Gaussian blur) + downsampling.

    Parameters:
        img: np.ndarray of shape (H, W, 3), dtype=float32 or float64
        factor: int, downsampling factor
        kernel_size: int, size of Gaussian blur kernel (odd)

    Returns:
        downsampled_img: np.ndarray of shape (H//factor, W//factor, 3)
    """
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply Gaussian blur to avoid aliasing
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Downsample by picking every 'factor' pixel
    downsampled = blurred[::factor, ::factor, :]

    return downsampled

def get_grad_world_gpu(image, rotation_matrix):
    
    image_cp = cp.asarray(image, dtype=cp.float16)

    # Compute Sobel gradients along x and y (axis=1 for x, axis=0 for y)
    gx = sobel(image_cp, axis=1, mode='nearest')
    gy = sobel(image_cp, axis=0, mode='nearest')

    # Compute mean across color channels if image has 3 channels
    if image_cp.ndim == 3:
        gx_1d = cp.mean(gx, axis=2, dtype = cp.float16)
        gy_1d = cp.mean(gy, axis=2, dtype = cp.float16)
    else:
        gx_1d = gx
        gy_1d = gy

    # Flatten gradients
    gx_1d_flat = gx_1d.ravel()
    gy_1d_flat = gy_1d.ravel()

    # Stack into (N, 3) gradient vectors (Z-component is zero)
    zeros = cp.zeros_like(gx_1d_flat, dtype = cp.float16)
    gxyz = cp.stack((gx_1d_flat, gy_1d_flat, zeros), axis=1)

    # Apply rotation
    rotation_matrix_cp = cp.asarray(rotation_matrix, dtype=cp.float16)
    gxyz_world = gxyz @ rotation_matrix_cp.T

    # Reshape to original image shape with 3 channels
    H, W = gx_1d.shape
    gxyz_world = gxyz_world.reshape((H, W, 3))

    return gxyz_world

def scale_intrinsics(K, scale):
    K_new = K.copy()
    K_new[0,0] /= scale   # fx
    K_new[1,1] /= scale   # fy
    K_new[0,2] /= scale   # cx
    K_new[1,2] /= scale   # cy
    K_new[2,2] = 1.0      # keep homogeneous coordinate
    return K_new

def similarityRGB(setA, setB):
    
    normA = cp.linalg.norm(setA, axis = -1)
    normB = cp.linalg.norm(setB, axis = -1)
    normDif = cp.linalg.norm(setA - setB, axis = -1)
    rgbSim = 1 - cp.abs(normDif) / (normA + normB + 1e-8)
    
    maskDark = (normA < 0.1) | (normB < 0.1)
    
    rgbSim[maskDark] = 0
        
    return rgbSim

def similarityGrad(setA, setB):

    normA = cp.linalg.norm(setA, axis = -1)
    normB = cp.linalg.norm(setB, axis = -1)
    dotProducts = cp.sum(setA * setB, axis = -1)
    cosSim = ((dotProducts/ (normA * normB + 1e-8))+1)/2
    normSim = 1 - cp.abs(normA - normB) / (normA + normB + 1e-8)
    combinedSim = cosSim * normSim
    
    return combinedSim