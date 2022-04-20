import numpy as np
from plyfile import PlyData, PlyElement
# import cv2
import open3d as o3d
import pickle
import os.path
import glob
# {"CalibrationInformation":{"Cameras":[{"Intrinsics":{"ModelParameterCount":14,"ModelParameters":[0.50127333402633667,0.50319623947143555,0.49252399802207947,0.49259248375892639,6.0841460227966309,4.3754911422729492,0.23618598282337189,6.4087333679199219,6.4182958602905273,1.2202920913696289,0,0,6.5029336838051677E-5,-5.5295702622970566E-5],"ModelType":"CALIBRATION_LensDistortionModelBrownConrady"},"Location":"CALIBRATION_CameraLocationD0","Purpose":"CALIBRATION_CameraPurposeDepth","MetricRadius":1.7399997711181641,"Rt":{"Rotation":[1,0,0,0,1,0,0,0,1],"Translation":[0,0,0]},"SensorHeight":1024,"SensorWidth":1024,"Shutter":"CALIBRATION_ShutterTypeUndefined","ThermalAdjustmentParams":{"Params":[0,0,0,0,0,0,0,0,0,0,0,0]}},{"Intrinsics":{"ModelParameterCount":14,"ModelParameters":[0.498662531375885,0.50206762552261353,0.47818887233734131,0.63736248016357422,0.58438080549240112,-2.4942491054534912,1.3602830171585083,0.46475523710250854,-2.3362464904785156,1.3011596202850342,0,0,-0.00017650700465310365,0.0006530816899612546],"ModelType":"CALIBRATION_LensDistortionModelBrownConrady"},"Location":"CALIBRATION_CameraLocationPV0","Purpose":"CALIBRATION_CameraPurposePhotoVideo","MetricRadius":0,"Rt":{"Rotation":[0.99999535083770752,-0.001910345396026969,-0.0023798577021807432,0.0021437217947095633,0.994754433631897,0.10226964205503464,0.0021720037329941988,-0.10227426886558533,0.994753897190094],"Translation":[-0.032022722065448761,-0.0020792526192963123,0.0039346059784293175]},"SensorHeight":3072,"SensorWidth":4096,"Shutter":"CALIBRATION_ShutterTypeUndefined","ThermalAdjustmentParams":{"Params":[0,0,0,0,0,0,0,0,0,0,0,0]}}],"InertialSensors":[{"BiasTemperatureModel":[-0.0039670444093644619,0,0,0,0.054306682199239731,0,0,0,-0.0012853745138272643,0,0,0],"BiasUncertainty":[9.9999997473787516E-5,9.9999997473787516E-5,9.9999997473787516E-5],"Id":"CALIBRATION_InertialSensorId_LSM6DSM","MixingMatrixTemperatureModel":[0.99807053804397583,0,0,0,0.0028721010312438011,0,0,0,-0.0012125088833272457,0,0,0,0.0028485429938882589,0,0,0,1.006335973739624,0,0,0,-0.0035347063094377518,0,0,0,-0.0012096856953576207,0,0,0,-0.0035556410439312458,0,0,0,1.0004042387008667,0,0,0],"ModelTypeMask":16,"Noise":[0.00095000001601874828,0.00095000001601874828,0.00095000001601874828,0,0,0],"Rt":{"Rotation":[-0.00045516181853599846,0.10896477103233337,-0.99404549598693848,-0.99999970197677612,0.00059037143364548683,0.00052260322263464332,0.00064380140975117683,0.9940454363822937,0.1089644655585289],"Translation":[0,0,0]},"SecondOrderScaling":[0,0,0,0,0,0,0,0,0],"SensorType":"CALIBRATION_InertialSensorType_Gyro","TemperatureBounds":[5,60],"TemperatureC":0},{"BiasTemperatureModel":[0.10213988274335861,0,0,0,0.032240699976682663,0,0,0,0.016003657132387161,0,0,0],"BiasUncertainty":[0.0099999997764825821,0.0099999997764825821,0.0099999997764825821],"Id":"CALIBRATION_InertialSensorId_LSM6DSM","MixingMatrixTemperatureModel":[1.0013891458511353,0,0,0,-2.2392507162294351E-5,0,0,0,-0.0022162268869578838,0,0,0,-2.2664989955956116E-5,0,0,0,0.9893454909324646,0,0,0,0.00042303174268454313,0,0,0,-0.0022265540901571512,0,0,0,0.000419893505750224,0,0,0,0.996744692325592,0,0,0],"ModelTypeMask":56,"Noise":[0.010700000450015068,0.010700000450015068,0.010700000450015068,0,0,0],"Rt":{"Rotation":[-0.0023786253295838833,0.10783781111240387,-0.99416565895080566,-0.9999966025352478,0.00082316645421087742,0.0024818656966090202,0.001086002797819674,0.99416816234588623,0.10783548653125763],"Translation":[-0.051182731986045837,0.0035288673825562,0.0016824802150949836]},"SecondOrderScaling":[0,0,0,0,0,0,0,0,0],"SensorType":"CALIBRATION_InertialSensorType_Accelerometer","TemperatureBounds":[5,60],"TemperatureC":0}],"Metadata":{"SerialId":"000573600112","FactoryCalDate":"1/1/2020 6:03:11 PM GMT","Version":{"Major":1,"Minor":2},"DeviceName":"AzureKinect-PV","Notes":"PV0_max_radius_invalid"}}}
'''
===== Device 0: 000573600112 ===== (Kinect Azure)
resolution width: 640
resolution height: 576
principal point x: 320.803894
principal point y: 334.772949
focal length x: 504.344574
focal length y: 504.414703
radial distortion coefficients:
k1: 6.084146
k2: 4.375491
k3: 0.236186
k4: 6.408733
k5: 6.418296
k6: 1.220292
center of distortion in Z=1 plane, x: 0.000000
center of distortion in Z=1 plane, y: 0.000000
tangential distortion coefficient x: -0.000055
tangential distortion coefficient y: 0.000065
metric radius: 0.000000
'''

# REALSENSE_D415_INTRINSICS = {"width": 640, "height": 640, "ppx": 307.155, "ppy": 243.155, "fx": 621.011, "fy": 621.011, "model": "Brown Conrady", "coeffs": [0, 0, 0, 0, 0]}
REALSENSE_D415_INTRINSICS = {"width": 1280, "height": 720, "ppx": 307.155, "ppy": 243.155, "fx": 621.011, "fy": 621.011, "model": "Brown Conrady", "coeffs": [0, 0, 0, 0, 0]}
KINECT_AZURE_INTRINSICS = {"width": 1080, "height": 720, "ppx": 320.803894, "ppy": 334.772949, "fx": 504.344574, "fy": 504.414703, "model": "Brown Conrady", "coeffs": [0, 0, -0.000055, 0.000065, 0]}

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

def save_projected(filename, image, depth, point_pairs, points, colours):
    '''
    Saves data from the output of project_2D
    '''
    filename = filename[:-4] + '.projected'
    # Check if filename exists. If it does not, create one
    if os.path.isfile(filename):
        print("File Exists")
        
    else:
        with open((filename), 'wb') as file:
            pickle.dump(image, file)
            pickle.dump(depth, file)
            pickle.dump(point_pairs, file)
            pickle.dump(points, file)
            pickle.dump(colours, file)
            file.close()

        print("Save Complete: " + filename)


def load_projected(filename):
    '''
    Loads data from the output of project_2D
    '''
    filename = filename.split('.')[0]
    filename = filename + '.projected'
    if os.path.isfile(filename):
        print("File Exists")
        with open((filename), 'rb') as file:
            image = pickle.load(file)
            print("33%")
            depth = pickle.load(file)
            print("66%")
            point_pairs = pickle.load(file)
            print("100%")
            points = pickle.load(file)
            colours = pickle.load(file)
    else:
        
        return False, None, None, None, None, None
    
    return True , image, depth, point_pairs, points, colours

def project_2D(intrinsics, plydata):
    '''
    This function takes 3D data and maps it to 2D. 
    '''
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

def project_all(folder_name):
    
    for file in glob.glob((folder_name+"*.ply")):
        loaded, image, depth, point_pairs, points, colours = load_projected(file)
        print(file)
        if not loaded:
            plydata = PlyData.read(file)
            image, depth, point_pairs = project_2D(KINECT_AZURE_INTRINSICS, plydata)
            cloud, points, colours = map_pairs_2D(image, point_pairs)
            save_projected(file, image, depth, point_pairs, points, colours)

def map_pairs_2D(image, point_pairs):
    # print(image)
    coordinates = get_valid_coordinates(image)
    # print(image.shape)
    points = []
    colours = []
    # length = len(coordinates)
    # print(point_pairs.keys())
    for index, pixel in enumerate(coordinates):
        # print((index + "/" + length ),end="\r")
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
    print("Mapping...")
    cloud = numpy_to_o3d(np_cloud_points=points,np_cloud_colors=colours)
    return cloud, points, colours

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

def numpy_to_o3d(np_cloud_points, np_cloud_colors=None, np_cloud_normals=None, swap_RGB=False):
    #create o3d pointcloud and assign it
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np_cloud_points)
    if np_cloud_colors is not None:
        if swap_RGB:
            R = np_cloud_colors[:,0].copy()
            # G = np_cloud_colors[:,1].copy()
            B = np_cloud_colors[:,2].copy()
            
            np_cloud_colors[:,0] = B
            np_cloud_colors[:,2] = R

        o3d_cloud.colors = o3d.utility.Vector3dVector(np_cloud_colors.astype(np.float) / 255.0)
    if np_cloud_normals is not None:
        o3d_cloud.normals = o3d.utility.Vector3dVector(np_cloud_normals)

    return o3d_cloud

def o3d_polygon(points, colour = [1,0,0]):
    # Connect points in sequence
    points = list(points)
    print("list", points)
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

    lineset_points = []
    lines = []
    colours = []
    spheres = []
    for index, point in enumerate(points):
        print(point)
        lineset_points.append([point[0], point[1], point[2]])
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

        sphere_translate = np.asarray(
                [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                [point[0], point[1], point[2], 1.0]])

        mesh_sphere.transform(sphere_translate)
        spheres.append(mesh_sphere)

        if index != len(points):
            lines.append([index, index+1])
            colours.append(colour)
    # point = np.asarray(points)
    print(points)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colours)
    
    # Line set is 1 shape and spheres is a list of shapes, we want a list of all objects
    # shapes = spheres.append(line_set)
    shapes = [line_set]
    shapes.extend(spheres)
    return shapes

def o3d_add_object(vis, objects):
    for object in objects:
        vis.add_geometry(object, reset_bounding_box=False)

def display_cloud(cloud, shapes=None):

    # if shapes is None:
    #     vis = o3d.visualization.draw_geometries([cloud])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)
    for object in shapes:  
        vis.add_geometry(object)
        # vis = o3d.visualization.draw_geometries(shapes)
    print("Returning visualizer", vis)
    return vis

def edit_cloud(cloud, shapes=None):
    vis = o3d.visualization.draw_geometries_with_editing([cloud])
    return vis
    # o3d.visualization.draw_geometries_with_editing([])

def get_3d_from_pairs(points_2D, point_pairs):
    points_3d = []
    for point in points_2D:
        if point in point_pairs.keys():
            points_3d.append(point_pairs[point][0])

    return points_3d

def get_distance(point_pairs, p1, p2):
    # print(p1, p2)
    points = np.array([p1, p2])
    # print(points)

    # cloud, points, colour = map_pairs_2D(points, point_pairs)

    return distance

    
if __name__ == "__main__":
    project_all("C:/Users/legom/Documents/Shared Folder/")
    ''' 
    plydata = PlyData.read("K:/Data/Data/Duck/Duckhouse/01_12_2022_19_54_23_760796.ply")
    image, depth, point_pairs = project_2D(REALSENSE_D415_INTRINSICS, plydata)
    print(image)
    cloud = map_pairs_2D(image, point_pairs)
    # cloud = project_3D(REALSENSE_D415_INTRINSICS, image, depth)
    o3d.visualization.draw_geometries([cloud])
    
    # np._replace_nan
    cv2.imshow("IMAGE",np.nan_to_num(image).astype(np.uint8))
    while True:
        cv2.waitKey(1)
    '''

    
    # cv2.imread()
    # print(point)
    # load_intrinsics("C:/Users/legom/Documents/GitHub/3DMeasure/data/Personal/out_1_intrinsics.txt")
    