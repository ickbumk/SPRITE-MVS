import numpy as np
import open3d as o3d
import re
import cv2
import glob
import cupy as cp
import plotly.graph_objects as go
from cupyx.scipy.ndimage import distance_transform_edt

def visualize(pts):
    
    if pts.shape[0] != 3:
        pts = pts.T

    # pts shape: (3, 1200)
    x, y, z = pts[0], pts[1], pts[2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=3)
    )])

    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()


def get_voxel_from_vertices(vertices, number = 100):
    xyz_min = np.min(vertices, axis = 0)
    xyz_max = np.max(vertices, axis = 0)

    dx,dy,dz = (xyz_max-xyz_min)/100
    
    x_mesh, y_mesh, z_mesh =np.linspace(xyz_min, xyz_max, num = number).T
    xx_mesh, yy_mesh, zz_mesh = np.meshgrid(x_mesh, y_mesh, z_mesh)

    #this is a 3 column vector, with number^3 rows
    voxels = np.vstack([xx_mesh.flatten(), yy_mesh.flatten(), zz_mesh.flatten()]).T

    return voxels, np.array([dx, dy, dz])

def nplist_to_cplist(nplist):
    '''
    Input : Numpy List
    Output : Cupy list (GPU)
    '''
    
    cplist = [cp.asarray(X) for X in nplist]
    return cplist

def get_random(vv, dv, n = 3):
    mag = np.random.normal(size = n)*np.linalg.norm(dv)
    uvec = np.random.randn(3, n)
    uvec /= np.linalg.norm(uvec, axis = 0)
    dvec = uvec*mag

    return vv + dvec.T

def get_hom(xyz):
    
    a,b = xyz.shape
    
    if a < b:
        xyz_hom = np.vstack([xyz, np.ones(b)[np.newaxis,:]])
    elif b < a:
        xyz_hom = np.hstack([xyz, np.ones(a)[:,np.newaxis]])

    return xyz_hom

def get_CV_matrices(projection):
    
    #This already provided T (camera center) in world coordinates
    K, R, C_hom, _,_,_,_ = cv2.decomposeProjectionMatrix(projection)
    
    C = C_hom[:3] / C_hom[3]
    
    T = -R@C
    
    return cp.array(K), cp.array(R), cp.array(T), cp.array(C)
    

def read_projection_txt(folder_path):
    file_name = sorted(glob.glob(folder_path+"/*.txt"))
    
    number_list = []
    projection_list = []
    
    for filepath in file_name:
        
        #Get the camera number
        match = re.search(r'pos_(\d+)\.txt', filepath)
        if match:
            number = match.group(1)        
            number_list.append(float(number))
            
        #Get the txt file
        
        with open(filepath, "r") as file:
            lines = []
            for line in file:
                p1,p2,p3,p4 = line.strip().split()
                
                lines.append(np.array([p1,p2,p3,p4], dtype = float))
            lines_array = np.vstack(lines)
            
            projection_list.append(lines_array)
                
    return number_list, projection_list

def read_pair(filename):

    '''
    Inputs: 
        filename = path to pair.txt

    Output:
        pair_list = list of file indices that has good match to the specific camera
    '''
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    pair_list = []
    num_entries = int(lines[0].strip())
    
    for i in range(num_entries):
        
        n_cam = int(lines[i*2+2].strip().split()[0])
        top10 = []    
        for j in range(n_cam):
            top10.append(lines[i*2+2].strip().split()[j*2+1])
    
        pair_list.append(top10)

    return np.array(pair_list)

def read_images(img_path, grayscale = False):
    img_filenames = sorted(glob.glob(img_path + '/*.png'))
    
    if not img_filenames:
        img_filenames = sorted(glob.glob(img_path + '/*.jpg'))
    img_list = []
    
        
    for img_filename in img_filenames:
        if grayscale:
            img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE) 
        else:
            img = cv2.imread(img_filename) 
        img_list.append(img)

    return img_list

def save_pointcloud_ply(points, colors=None, filename="output.ply", binary=True):


    # --- Validate input ---
    if points.dtype != np.float64:
        raise TypeError("Expecting float64 for points")
    if colors is not None and colors.dtype != np.float64:
        raise TypeError("Expecting float64 for colors")

    print(f"Saving point cloud to {filename}...")

    # --- Build point cloud ---
    pcd = o3d.geometry.PointCloud()
    points = np.ascontiguousarray(points)
    pcd.points = o3d.utility.Vector3dVector(points)

    # Optional colors
    if colors is not None:
        if colors.shape != points.shape:
            raise ValueError("Colors must be an Nx3 array matching the points.")

        # Convert 0–255 → 0–1 if needed
        if colors.max() > 1.0:
            colors = colors / 255.0

        colors = np.ascontiguousarray(colors)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # ASCII vs binary
    write_option = o3d.io.write_point_cloud

    success = write_option(
        filename,
        pcd,
        write_ascii=(not binary),
        compressed=False
    )

    if success:
        print(f"Successfully saved: {filename}")
    else:
        print("Failed to save PLY file!")


def fill_nan_nearest(depth, mask):
    _, idx = distance_transform_edt(mask, return_indices=True)
    filled = depth.copy()
    filled[mask] = depth[tuple(idx[:, mask])]
    return filled

