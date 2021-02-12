import numpy as np
from plyfile import PlyData, PlyElement
import cv2
import open3d as o3d

REALSENSE_D415_INTRINSICS = {"width": 640, "height": 640, "ppx": 307.155, "ppy": 243.155, "fx": 621.011, "fy": 621.011, "model": "Brown Conrady", "coeffs": [0, 0, 0, 0, 0]}

def map_3D_to_2D(intrinsics, point3D):

    point2D = np.zeros((2,), dtype=int)

    x = point3D[0]/ point3D[2]
    y = point3D[1] / point3D[2]

    if (intrinsics['model'] == 'Brown Conrady'):
        r2 = x * x + y * y
        f = 1 + intrinsics['coeffs'][0] * r2 + intrinsics['coeffs'][1] * r2*r2 + intrinsics['coeffs'][4] * r2*r2*r2

        xf = x * f
        yf = y * f

        dx = xf + 2 * intrinsics['coeffs'][2] * x*y + intrinsics['coeffs'][3] * (r2 + 2 * x*x)
        dy = yf + 2 * intrinsics['coeffs'][3] * x*y + intrinsics['coeffs'][2] * (r2 + 2 * y*y)

        x = dx
        y = dy
    
    
    point2D[0] = x * intrinsics['fx'] + intrinsics['ppx']
    point2D[1] = y * intrinsics['fy'] + intrinsics['ppy']

    return point2D



def map_2D_to_3D(intrinsics, pixel, depth):
    
    point3D = np.zeros((3,), dtype=np.float)

    x = (pixel[0] - intrinsics['ppx']) / intrinsics['fx']
    y = (pixel[1] - intrinsics['ppy']) / intrinsics['fy']
    if (intrinsics['model'] == 'Brown Conrady'):
    
        r2 = x * x + y * y
        f = 1 + intrinsics['coeffs'][0] * r2 + intrinsics['coeffs'][1] * r2 * r2 + intrinsics['coeffs'][4] * r2 * r2 * r2
        ux = x * f + 2 * intrinsics['coeffs'][2] * x * y + intrinsics['coeffs'][3] * (r2 + 2 * x * x)
        uy = y * f + 2 * intrinsics['coeffs'][3] * x * y + intrinsics['coeffs'][2] * (r2 + 2 * y * y)

        x = ux
        y = uy
    
    # print(x,y,depth)
    point3D[0] = depth * x
    point3D[1] = depth * y
    point3D[2] = depth

    return point3D

def project_2D(intrinsics, plydata):
    # plydata = PlyData.read(f)
    num_verts = plydata['vertex'].count
    vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
    vertices[:,0] = plydata['vertex'].data['x']
    vertices[:,1] = plydata['vertex'].data['y']
    vertices[:,2] = plydata['vertex'].data['z']
    vertices[:,3] = plydata['vertex'].data['red']
    vertices[:,4] = plydata['vertex'].data['green']
    vertices[:,5] = plydata['vertex'].data['blue']

    # image = np.zeros((intrinsics['height'], intrinsics['width'], 3), np.uint8)
    image = np.full((intrinsics['height'], intrinsics['width'], 3), np.nan)
    # image.fill(256)
    # print(image)
    point_pairs = {}
    for index in range(num_verts):
        point = (plydata['vertex'].data['x'][index], plydata['vertex'].data['y'][index], plydata['vertex'].data['z'][index])
        colour = (plydata['vertex'].data['blue'][index],plydata['vertex'].data['green'][index],plydata['vertex'].data['red'][index])

        projected_point = map_3D_to_2D(intrinsics, point)
        # print(projected_point)
        #Proper Orientation
        # image[projected_point[1],-projected_point[0]] = colour
        image[projected_point[1],projected_point[0]] = colour
        point_pairs[(projected_point[0], projected_point[1])] = (point, colour)
        # print(image[projected_point[0],projected_point[1]])
    # print(image)
    print("Project to 2D Complete")
    return image, plydata['vertex'].data['z'], point_pairs

def project_3D(intrinsics, image, depth):

    # image = np.nan_to_num(image).astype(np.uint8)
    image = np.nan_to_num(image, nan=-1)
    indices = np.where(image != [-1])
    coordinates = zip(indices[1], indices[0])
    print(coordinates)
    image = image.flatten()
    image = np.delete(image, np.where(image == -1))
    colours = np.reshape(image,(depth.shape[0],3))

    points = []
    print(len(depth))
    print(len(list(coordinates)))
    for index, point in enumerate(coordinates):
        # print(point)
    # REALSENSE_D415_INTRINSICS["width"]
    # depth = np.reshape(depth,))
    
    # for x in range(image.shape[0]):
    # # for ix,iy in np.ndindex(image.shape):
    #     colour = image[x].astype(np.float)
    #     print("color", colour)
        
    #     if colour[0] != -1:
    #         # print(colour, x, y, depth[x+y])
        # print(index)
        if index == len(depth):
            break
        point3D = map_2D_to_3D(intrinsics, (point[0], point[1]), depth[index])
        # print(point3D)
        # colours.append(colour)
        points.append(point3D)
            
    colours = np.asarray(colours).astype(np.uint8)
    points = np.asarray(points)
    # print(points)
    cloud = numpy_to_o3d(np_cloud_points=points, np_cloud_colors=colours)

    return cloud

def map_pairs_2D(image, point_pairs):
    coordinates = get_valid_coordinates(image)
    # print(image.shape)
    points = []
    colours = []
    print(point_pairs.keys())
    for pixel in coordinates:
        try:
            point, colour = point_pairs[pixel]
            points.append(point)
            colours.append(colour)
            # print("Adding ", pixel, colour)
        except:
            # print("Skipping", pixel)
            continue

    colours = np.asarray(colours).astype(np.uint8)
    points = np.asarray(points)
    cloud = numpy_to_o3d(np_cloud_points=points,np_cloud_colors=colours)
    return cloud

def strip_null(image):
    image = np.nan_to_num(image, nan=-1)
    image = image.flatten()
    image = np.delete(image, np.where(image == -1))
    image = np.reshape(image,((int(image.shape[0]/3)),3))
    return image

def get_valid_coordinates(image):
    image = np.nan_to_num(image, nan=-1)
    indices = np.where(image != [-1])
    coordinates = zip(indices[1], indices[0])
    return coordinates

def load_intrinsics(filename):
    '''
    Example:
    width: 640, height: 480, ppx: 307.155, ppy: 243.155, fx: 621.011, fy: 621.011, model: Brown Conrady, coeffs: [0, 0, 0, 0, 0]
    '''
    file = open(filename, "r")
    intrinsic_str = file.readline()
    intrinsics_list = {}
    intrinsics = intrinsic_str.split(", ")
    
    # print(intrinsics)peint
    # for index, intrinsic in enumerate(intrinsics):
        # print(intrinsic)
        # value = intrinsic.split(": ")
        # # print(value[0])
        # if value[0] is not "coeffs":
        #     intrinsics_list[value[0]] = value[1]
        #     print(value)


    # intrinsics = {"width":, "height"}

    # project_2D
def o3d_to_numpy(o3d_cloud):
    """
    Converts open3d pointcloud to numpy array
    """
    np_cloud = np.asarray(o3d_cloud.points)
    return np_cloud

def numpy_to_o3d(np_cloud_points, np_cloud_colors=None, np_cloud_normals=None):
    #create o3d pointcloud and assign it
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np_cloud_points)
    if np_cloud_colors is not None:
        o3d_cloud.colors = o3d.utility.Vector3dVector(np_cloud_colors.astype(np.float) / 255.0)
    if np_cloud_normals is not None:
        o3d_cloud.normals = o3d.utility.Vector3dVector(np_cloud_normals)

    return o3d_cloud

def display_cloud(cloud):
    o3d.visualization.draw_geometries([cloud])

def edit_cloud(cloud):
    o3d.visualization.draw_geometries_with_editing([cloud])
    # o3d.visualization.draw_geometries_with_editing([])

        
if __name__ == "__main__":
    plydata = PlyData.read("C:/Users/legom/Documents/GitHub/RemoteAnimalScan/ras/CreateDataset/816612061344_no1.ply")
    image, depth, point_pairs = project_2D(REALSENSE_D415_INTRINSICS, plydata)
    cloud = map_pairs_2D(image, point_pairs)
    # cloud = project_3D(REALSENSE_D415_INTRINSICS, image, depth)
    o3d.visualization.draw_geometries([cloud])
    
    # np._replace_nan
    cv2.imshow("IMAGE",np.nan_to_num(image).astype(np.uint8))
    while True:
        cv2.waitKey(1)

    # cv2.imread()
    
    
    # print(point)
    # load_intrinsics("C:/Users/legom/Documents/GitHub/3DMeasure/data/Personal/out_1_intrinsics.txt")
    