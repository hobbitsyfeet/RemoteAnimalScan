import numpy as np
from plyfile import PlyData, PlyElement
import cv2
import open3d as o3d
import pickle
import os.path
import glob
from scipy.spatial.transform import Rotation as R

import json
import uuid

import random
import concurrent.futures
import multiprocessing

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
KINECT_AZURE_INTRINSICS = {"width": 1280, "height": 720, "ppx": 320.803894, "ppy": 334.772949, "fx": 504.344574, "fy": 504.414703, "model": "Brown Conrady", "coeffs": [0, 0, -0.000055, 0.000065, 0]}

def map_3D_to_2D(intrinsics, point3D):
    '''
     [u]   [fx  0  cx][r11 r12 r13 t1][X]
    s[v] = [0   fy cy][r21 r22 r23 t2][Y]
     [1]   [0   0   1][r31 r32 r33 t3][Z]
                                      [1]

    x,y,z are 3D coordiantes
    u,v are coordinates of projection in pixels
    A is camera Matrix (rotations and transformations)
    cx,cy are principal point (camera center)
    fx and fy are focal lengths in pixel units

    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20projectPoints(InputArray%20objectPoints,%20InputArray%20rvec,%20InputArray%20tvec,%20InputArray%20cameraMatrix,%20InputArray%20distCoeffs,%20OutputArray%20imagePoints,%20OutputArray%20jacobian,%20double%20aspectRatio)
    
        [x] =  [X]
        [y] = R[Y] + t
        [z] =  [Z]

        x' = x/z
        y' = y/z

        NOTE: k1-6 are radial distortion coefficients, p1 and p2 are tangential distortion
                if any 
        x'' = x' (1 + k1 * r^2+k2 + k3*r^4 + k3*r^6)/ + 2*p1*x'*y' + p2(r^2 + (2x')^2)
                 (1 + k4 * r^2+k5 + k3*r^4 + k6*r^6)

        y'' = y' (1 + k1 * r^2+k2 + k3*r^4 + k3*r^6)/ + p1(r^2 + (2y')^2) + 2*p2 * x' * y'
                 (1 + k4 * r^2+k5 + k3*r^4 + k6*r^6)

        r^2 = x'^2 + y'^2
        u = fx * x'' + cx
        v = fy * y'' + cy

    NOTE: Use this to multiply transformation across all the points
    >>> pts_einsum = np.einsum("ij,kj->ik", pts, transform_matrix)
    >>> np.allclose(pts_brute, pts_einsum)
    https://stackoverflow.com/questions/26289972/use-numpy-to-multiply-a-matrix-across-an-array-of-points
    '''
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

def map_3D_to_2D_parallel(intrinsics, points3D):
    num_points = len(points3D)
    points2D = np.zeros((num_points, 2), dtype=int)

    def process_point(index):
        point3D = points3D[index]
        x = point3D[0] / point3D[2]
        y = point3D[1] / point3D[2]

        if intrinsics['model'] == 'Brown Conrady':
            r2 = x * x + y * y
            f = 1 + intrinsics['coeffs'][0] * r2 + intrinsics['coeffs'][1] * r2 * r2 + intrinsics['coeffs'][4] * r2 * r2 * r2

            xf = x * f
            yf = y * f

            dx = xf + 2 * intrinsics['coeffs'][2] * x * y + intrinsics['coeffs'][3] * (r2 + 2 * x * x)
            dy = yf + 2 * intrinsics['coeffs'][3] * x * y + intrinsics['coeffs'][2] * (r2 + 2 * y * y)

            x = dx
            y = dy

        point2D = np.zeros((2,), dtype=int)
        point2D[0] = x * intrinsics['fx'] + intrinsics['ppx']
        point2D[1] = y * intrinsics['fy'] + intrinsics['ppy']
        return point2D

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_point, i) for i in range(num_points)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            points2D[i] = future.result()

    return points2D



def matrix_3D_to_2D(points3D, colour, intrinsics):
    # pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    # image = np.full((intrinsics['height'], intrinsics['width'], 3), np.nan)
    image = np.full((5000, 5000, 3), np.nan)
    camera = np.matrix([
                        [ intrinsics['fx'],    0,                    intrinsics['ppx']  ], 
                        [ 0,                   intrinsics['fy'],     intrinsics['ppy']  ],
                        [ 0,                   0,                    1                  ],
                        ])
    dist = np.asfarray(intrinsics['coeffs'])
    '''
    
    #NOTE:
    Need Rvec and Tvec from https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1427
    Device[Instance].GetCalibration().ColorCameraCalibration.Extrinsics.Translation/Rotation
    rvec = np.zeros(3, dtype=np.float)
    tvec = np.array([0, 0, 1], dtype=np.float)
    '''

    projected = cv2.projectPoints(points3D,
                      rvec=None, 
                      tvec=None,
                      cameraMatrix=camera,
                      distCoeffs=dist)
    
    pixels = projected[0]
    pixels = pixels.astype(np.int)
    # print(pixels)
    # print(pixels.shape)
    # pixels.reshape(intrinsics['height'], intrinsics['width'], 1)
    min = np.amin(pixels, axis=0)
    print(min)
    for index, pixel in enumerate(pixels):
        pixel = pixel[0]
        image[pixel[1]+ abs(min[0][1]), pixel[0]+abs(min[0][0])] = colour[index]


    return image

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
            # pickle.dump(inverse_pairs, file)
            file.close()

        print("Save Complete: " + filename)


def load_projected(filename, progressbar=None):
    '''
    Loads data from the output of project_2D
    '''
    filename = filename.split('.')[0]
    filename = filename + '.projected'
    
    # if progressbar:
        # progressbar.setFormat('Loading Projected Data...')

    if os.path.isfile(filename):
        with open((filename), 'rb') as file:

            if progressbar is not None:
                progressbar.setValue(0)
            print("0%")
            image = pickle.load(file)

            if progressbar is not None:
                progressbar.setValue(10)
            print("20%")

            depth = pickle.load(file)
            if progressbar is not None:
                progressbar.setValue(20)
            print("40%")

            point_pairs = pickle.load(file)
            if progressbar is not None:
                progressbar.setValue(30)
            print("60%")

            points = pickle.load(file)
            if progressbar is not None:
                progressbar.setValue(40)
            print("80%")

            colours = pickle.load(file)
            if progressbar is not None:
                progressbar.setValue(50)
            print("100%")

    else:
        return False, None, None, None, None, None #, None
    
    return True , image, depth, point_pairs, points, colours

def project_2D(intrinsics, plydata, progress=None):
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
    inverse_pairs = {}
    for index in range(num_verts):
        point = (plydata['vertex'].data['x'][index], plydata['vertex'].data['y'][index], plydata['vertex'].data['z'][index])
        colour = (plydata['vertex'].data['blue'][index],plydata['vertex'].data['green'][index],plydata['vertex'].data['red'][index])

        # projected_point = map_3D_to_2D(intrinsics, point)
        projected_point = map_3D_to_2D_parallel(intrinsics, point)
        # print(projected_point)
        #Proper Orientation
        # image[projected_point[1],-projected_point[0]] = colour
        image[projected_point[1],projected_point[0]] = colour
        point_pairs[(projected_point[0], projected_point[1])] = (point, colour)
        inverse_pairs[point] = [(projected_point[0], projected_point[1])]
        if progress is not None:
            # print(int((index/num_verts)*100))
            progress.window().processEvents()
            progress.setValue(int((index/num_verts)*100))
        # print(image[projected_point[0],projected_point[1]])
    # print(image)
    print("Project to 2D Complete")
    return image, plydata['vertex'].data['z'], point_pairs

def json_dict_to_numpy(json_dict):
    np_points = np.asarray(json_dict['points'])
    np_colours = np.asarray(json_dict['colours'])

    return np_points, np_colours

def json_dict_to_vert(json_dict):
    """
    converts json data to numpy vertices used in project_2D_multithread (exclusively)
    """
    num_verts = len(json_dict['points'])
    vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
    points, colours = json_dict_to_numpy(json_dict)
    
    vertices[:, 0] = points[:, 0]
    vertices[:, 1] = points[:, 1]
    vertices[:, 2] = points[:, 2]
    vertices[:, 3] = colours[:, 0]
    vertices[:, 4] = colours[:, 1]
    vertices[:, 5] = colours[:, 2]

    return vertices
    

def project_2D_multithread(intrinsics, data, progress=None, num_threads=None, dtype="ply"):
    vertices = None
    num_verts = 0
    if dtype == "ply":
        num_verts = data['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = data['vertex'].data['x']
        vertices[:, 1] = data['vertex'].data['y']
        vertices[:, 2] = data['vertex'].data['z']
        vertices[:, 3] = data['vertex'].data['red']
        vertices[:, 4] = data['vertex'].data['green']
        vertices[:, 5] = data['vertex'].data['blue']
    
    if dtype == "json":
        vertices = json_dict_to_vert(data)
        num_verts = vertices.shape[0]

    image = np.full((intrinsics['height'], intrinsics['width'], 3), np.nan)
    point_pairs = {}
    inverse_pairs = {}

    if num_threads is None:
        
        num_threads = max(1, multiprocessing.cpu_count() - 1)
        print("Number of threads:", num_threads)
    segment_size = num_verts // num_threads

    def process_segment(start_index, end_index):
        print("Starting segment:", start_index, ":", end_index)
        for index in range(start_index, end_index):
            point = (
                vertices[index, 0],
                vertices[index, 1],
                vertices[index, 2]
            )
            colour = (
                vertices[index, 3],
                vertices[index, 4],
                vertices[index, 5]
            )

            projected_point = map_3D_to_2D(intrinsics, point)
            image[projected_point[1], projected_point[0]] = colour
            point_pairs[(projected_point[0], projected_point[1])] = (point, colour)
            inverse_pairs[point] = [(projected_point[0], projected_point[1])]

            if progress is not None:
                progress.window().processEvents()
                progress.setValue(int((index / num_verts) * 100))
        print("Ending segment:", start_index, ":", end_index)

    # Use a ThreadPoolExecutor to execute the segment processing in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_threads):
            start_index = i * segment_size
            end_index = start_index + segment_size if i < num_threads - 1 else num_verts
            futures.append(executor.submit(process_segment, start_index, end_index))

    # Wait for all futures to complete
    concurrent.futures.wait(futures)

    print("Project to 2D Complete")
    return image, vertices[:, 2], point_pairs

# def project_2D_multithread(intrinsics, plydata, progress=None):
#     num_verts = plydata['vertex'].count
#     vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
#     vertices[:, 0] = plydata['vertex'].data['x']
#     vertices[:, 1] = plydata['vertex'].data['y']
#     vertices[:, 2] = plydata['vertex'].data['z']
#     vertices[:, 3] = plydata['vertex'].data['red']
#     vertices[:, 4] = plydata['vertex'].data['green']
#     vertices[:, 5] = plydata['vertex'].data['blue']

#     image = np.full((intrinsics['height'], intrinsics['width'], 3), np.nan)
#     point_pairs = {}
#     inverse_pairs = {}

#     def process_vertex(index):
#         point = (
#             plydata['vertex'].data['x'][index],
#             plydata['vertex'].data['y'][index],
#             plydata['vertex'].data['z'][index]
#         )
#         colour = (
#             plydata['vertex'].data['blue'][index],
#             plydata['vertex'].data['green'][index],
#             plydata['vertex'].data['red'][index]
#         )

#         projected_point = map_3D_to_2D(intrinsics, point)
#         image[projected_point[1], projected_point[0]] = colour
#         point_pairs[(projected_point[0], projected_point[1])] = (point, colour)
#         inverse_pairs[point] = [(projected_point[0], projected_point[1])]

#         if progress is not None:
#             progress.window().processEvents()
#             progress.setValue(int((index / num_verts) * 100))

#     # Use a ThreadPoolExecutor to execute the vertex processing in parallel
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_vertex, index) for index in range(num_verts)]

#     # Wait for all futures to complete
#     concurrent.futures.wait(futures)

#     print("Project to 2D Complete")
#     return image, plydata['vertex'].data['z'], point_pairs

def project_3D(intrinsics, image, depth):

    # image = np.nan_to_num(image).astype(np.uint8)
    image = np.nan_to_num(image, nan=-1)
    indices = np.where(image != [-1])
    coordinates = zip(indices[1], indices[0])

    image = image.flatten()
    image = np.delete(image, np.where(image == -1))
    colours = np.reshape(image,(depth.shape[0],3))

    points = []
    # print(len(depth))
    # print(len(list(coordinates)))
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
        # print(file)
        if not loaded:
            plydata = PlyData.read(file)
            image, depth, point_pairs, points, colours = project_2D(KINECT_AZURE_INTRINSICS, plydata)
            cloud, points, colours = map_pairs_2D(image, point_pairs)
            save_projected(file, image, depth, point_pairs,  points, colours)

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

    if isinstance(np_cloud_points, list):
        np_cloud_points = np.array(np_cloud_points)
    
    if isinstance(np_cloud_colors, list):
        np_cloud_colors = np.array(np_cloud_colors)

    if isinstance(np_cloud_normals, list):
        np_cloud_normals = np.array(np_cloud_normals)

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

def o3d_box(location, size=(3,3,3), colour=(1,0,0)):
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=size[0],
                                                    height=size[1],
                                                    depth=size[2])
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([colour[0], colour[1], colour[2]])
    mesh_box.translate((
                        (location[0] - (size[0]/2)), 
                        (location[1] - (size[1]/2)), 
                        (location[2] - (size[2]/2))
                        ))
    return mesh_box

def o3d_polygon(points, colour = [1,0,0]):
    # Connect points in sequence
    points = list(points)
    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    # mesh_sphere.compute_vertex_normals()
    # mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

    lineset_points = []
    lines = []
    colours = []
    boxes = []
    # spheres = []
    for index, point in enumerate(points):
        # print(point)
        lineset_points.append([point[0], point[1], point[2]])

        box = o3d_box([point[0], point[1], point[2]], colour=(0,1,0))
        boxes.append(box)
        # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        # mesh_sphere.compute_vertex_normals()
        # mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

        # sphere_translate = np.asarray(
        #         [[0, 0, 0, 0],
        #          [0, 0, 0, 0],
        #          [0, 0, 0, 0],
        #         [point[0], point[1], point[2], 1.0]])

        # mesh_sphere.transform(sphere_translate)
        # spheres.append(mesh_sphere)
        if index == 0 or index == len(points):
            lines.append([index, index])
            colours.append(colour)

        if index != len(points):
            lines.append([index, index+1])
            colours.append(colour)
    # point = np.asarray(points)
    # print(points)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colours)
    
    # Line set is 1 shape and spheres is a list of shapes, we want a list of all objects
    # shapes = spheres.append(line_set)
    shapes = [line_set]
    shapes = shapes + boxes
    return shapes

def o3d_add_object(vis, objects):
    for object in objects:
        vis.add_geometry(object, reset_bounding_box=False)

def display_cloud(cloud, shapes=None):

    # if shapes is None:
    #     vis = o3d.visualization.draw_geometries([cloud])
    vis = o3d.visualization.Visualizer()
    # vis = o3d.visualization.VisualizerWithVertexSelection()

    vis.create_window()

    vis.add_geometry(cloud)
    for object in shapes:  
        vis.add_geometry(object)
        # vis = o3d.visualization.draw_geometries(shapes)
    # print("Returning visualizer", vis)
    return vis

def get_global_filename(folder, filename):
        print(str((folder + "/" + filename)))
        return str((folder + "/" + filename))

def edit_cloud(cloud, shapes=None):
    vis = o3d.visualization.draw_geometries_with_editing([cloud])
    return o3d
    # o3d.visualization.draw_geometries_with_editing([])

def get_3d_from_pairs(points_2D, point_pairs):
    points_3d = []
    for point in points_2D:
        if point in point_pairs.keys():
            points_3d.append(point_pairs[point][0])

    return points_3d

def get_distance(p1, p2):
    '''
    requires projected points in 3D space'''
    distance = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)**(1/2)
    return distance

def get_distance_2D(point_pairs, pixel_1, pixel_2):
    distance = 0
    if pixel_1 in point_pairs.keys() and pixel_2 in point_pairs.keys():
        p1, colour = point_pairs[pixel_1]
        p2, colour = point_pairs[pixel_2]
        distance = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)**(1/2)
    else:
        print("Point does not exist to measure")
    return distance

def get_total_distance(point_pairs, points):
    '''
    returns total distance of 2d points

    Points is the polygon.points format
    '''
    total_distance = 0
    for index, point in enumerate(points):
        if index+1 < len(points):
            #pixel coordinates
            p1 = points[index]
            p2 = points[index+1]
            try:
                # pixels mapped to depth coordinates (if they are valid)
                if p1 in point_pairs.keys() and p2 in point_pairs.keys():
                    total_distance += get_distance_2D(point_pairs, p1, p2)
            except Exception as e:
                print(e)
                print("Could not grab point pairs")
                return 0
    return total_distance

def write_o3d_ply(filename, np_points, np_colours, labels=None):
    filename = filename.split(".")[0] + ".ply"
    if labels is None:
        cloud = numpy_to_o3d(np_points, np_colours)
        o3d.io.write_point_cloud(filename, cloud)

    if labels is not None:
        write_ply_labels(filename, np_points, np_colours)



def write_ply_labels(filename, xyz, rgb, label_type, labels, ids):
    # labels = range(len(set(labels)))

    # https://github.com/loicland/superpoint_graph/blob/a622b8bcdceb4a6749d36cbebc839637fa5145b0/partition/provider.py#L472
    """write into a ply file. include the label"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1')
            , ('blue', 'u1'), ('label_type', 'u1'), ('label', 'u1'), ('id', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = label_type
    vertex_all[prop[7][0]] = labels
    vertex_all[prop[8][0]] = ids
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)

    ply.write(filename)
    print(filename)

# def apply_labels_to_ply(filename, labels):
    

#     ply = PlyData.read(filename)
#     vertex = ply.elements[0]
#     colour = ply.elements[1]

#     # Create the new vertex data with appropriate dtype
#     label = np.empty(len(vertex.data), vertex.data.dtype.descr + [('label', 'i4')])
#     for name in vertex.data.dtype.fields:
#         label[name] = vertex[name]
#     label['label'] = 0

#     # Recreate the PlyElement instance
#     vertex = PlyElement.describe(label, 'vertex')
#     # Recreate the PlyData instance
#     ply = PlyData([vertex, colour], text=True)
#     ply.write('test.ply')

def convert_list_to_int(string_list):
    # Create an empty dictionary
    mapping = {}
    # Initialize a counter
    counter = 0
    # Iterate over the list of strings
    for string in string_list:
        # Check if the string is already in the dictionary
        if string not in mapping:
            # If not, add it to the dictionary and increment the counter
            mapping[string] = counter
            counter += 1
    # Create an empty list to store the integers
    int_list = []
    # Iterate over the list of strings again
    for string in string_list:
        # Lookup the integer value in the dictionary and append it to the list
        int_list.append(mapping[string])
    # Return the list of integers
    return int_list, mapping

def load_json_pointcloud(filename):
    """
    loads in json data and formats it into a dictionary
    keys = points, colours, labels, label_id, label_types
    """
    with open(filename, 'r') as json_file:
        pointcloud_and_labels = json.load(json_file)

        # Handle non-compatible datatypes
        pointcloud_and_labels['points'] = list(map(eval, pointcloud_and_labels['points']))
        pointcloud_and_labels['colours'] = list(map(eval, pointcloud_and_labels['colours']))
        pointcloud_and_labels['label_id'] = list(map(uuid.UUID, pointcloud_and_labels['label_id']))

    return pointcloud_and_labels


def compile_json_to_hdf(self, json_filelist):
    for json_file in json_filelist:
        json_data = self.load_json(json_file)
        cloud = json_dict_to_o3d(json_data)
        
    pass

def json_dict_to_o3d(pointcloud_dict):
    return numpy_to_o3d(pointcloud_dict['points'], pointcloud_dict['colours'])
    

def rgb_unscale(rgb):
    """
    takes scaled rgb between 0 and 1 and returns the same in 0-255
    """
    r = rgb[0] * 255
    g = rgb[1] * 255
    b = rgb[2] * 255

    return (r,g,b)

def random_sample_indices(sample_size, seed=42):
    # Assume numpy list
    random.seed(seed)
    indices = [i for i in range(sample_size)]
    sample_indices = random.sample(indices, k=sample_size)
    return sample_indices

def data_indices(data_list, target_value):
    return np.where(np.array(data_list) == target_value)[0]


def select_indices(data_list, indices_list):
    # Extract data at the given indices
    data_list = np.array(data_list)
    return data_list[indices_list]

def select_dict_indices(data_dict, indices):
    '''
    selects indices all for each dictionary key (each of the values in dict must be same length)
    '''
    sampled_dict = {}
    for key in data_dict.keys():
        sampled_dict[key] = select_indices(data_dict[key])
    return sampled_dict


def generate_indices(data_list):
    return range(0, len(data_list))

def unscale_rgb_np(numpy_array):
    return numpy_array*255


if __name__ == "__main__":
    #project_all("C:/Users/legom/Documents/Shared Folder/")
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
    print("Reading")
    plydata = PlyData.read("K:/Data/Data/Duck/Duckhouse/01_12_2022_19_54_23_760796.ply")
    pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    colour = np.vstack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T
    print("Matrix Projection")
    points = matrix_3D_to_2D(pts, colour,  KINECT_AZURE_INTRINSICS)
    print(points)

    cv2.imshow("image", points)
    cv2.waitKey(0)

    
    # cv2.imread()
    # print(point)
    # load_intrinsics("C:/Users/legom/Documents/GitHub/3DMeasure/data/Personal/out_1_intrinsics.txt")
    