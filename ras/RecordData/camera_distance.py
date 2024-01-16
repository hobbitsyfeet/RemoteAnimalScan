


# Calculate LUX
# https://www.atecorp.com/atecorp/media/pdfs/data-sheets/Tektronix-J16_Application.pdf


'''
Kinect Azure Sensor Error per lux and sensor info found at:
https://ieeexplore.ieee.org/document/8310200

https://static1.squarespace.com/static/551ecf1ae4b0b101cf72bfa3/t/55412684e4b0512f43caa5de/1430333060072/Resolution_calculation.pdf
'''
'''
### TABLE A ### 

Mode 	                    Resolution 	FoI 	    FPS 	        Operating range* 	Exposure time
NFOV unbinned 	            640x576 	75°x65° 	0, 5, 15, 30 	    0.5 - 3.86 m 	12.8 ms
NFOV 2x2 binned (SW) 	    320x288 	75°x65° 	0, 5, 15, 30 	    0.5 - 5.46 m 	12.8 ms
WFOV 2x2 binned 	        512x512 	120°x120° 	0, 5, 15, 30 	    0.25 - 2.88 m 	12.8 ms
WFOV unbinned 	            1024x1024 	120°x120° 	0, 5, 15 	        0.25 - 2.21 m 	20.3 ms
Passive IR 	                1024x1024 	N/A 	    0, 5, 15, 30 	    N/A 	        1.6 ms

*15% to 95% reflectivity at 850nm, 2.2 μW/cm2/nm, random error std. dev. ≤ 17 mm, typical systematic error < 11 mm + 0.1% of distance without multi-path interference. Depth may be provided outside of the operating range indicated above. It depends on an object's reflectivity.
Color camera supported operating modes

Azure Kinect DK includes an OV12A10 12MP CMOS sensor rolling shutter sensor. The native operating modes are listed below:
RGB Camera Resolution   (HxV) 	            Aspect Ratio 	Format Options 	Frame Rates (FPS) 	Nominal FOV (HxV)(post-processed)
3840x2160 	16:9 	    MJPEG 	            0, 5, 15, 30 	                                    90°x59°
2560x1440 	16:9 	    MJPEG 	            0, 5, 15, 30 	                                    90°x59°
1920x1080 	16:9 	    MJPEG 	            0, 5, 15, 30 	                                    90°x59°
1280x720 	16:9 	    MJPEG/YUY2/NV12 	0, 5, 15, 30 	                                    90°x59°
4096x3072 	4:3 	    MJPEG 	            0, 5, 15 	                                        90°x74.3°
2048x1536 	4:3 	    MJPEG 	            0, 5, 15, 30 	                                    90°x74.3°

The RGB camera is USB Video class-compatible and can be used without the Sensor SDK. The RGB camera color space: BT.601 full range [0..255]. The MJPEG chroma sub-sampling is 4:2:2.

Note

The Sensor SDK can provide color images in the BGRA pixel format. This is not a native mode supported by the device and causes additional CPU load when used. The host CPU is used to convert from MJPEG images received from the device.
RGB camera exposure time values

### TABLE B ###
Below is the mapping for the acceptable RGB camera manual exposure values:
exp 	2^exp 	50Hz 	60Hz
-11 	488 	500 	500
-10 	977 	1250 	1250
-9 	1953 	2500 	2500
-8 	3906 	10000 	8330
-7 	7813 	20000 	16670
-6 	15625 	30000 	33330
-5 	31250 	40000 	41670
-4 	62500 	50000 	50000
-3 	125000 	60000 	66670
-2 	250000 	80000 	83330
-1 	500000 	100000 	100000
0 	1000000 	120000 	116670
1 	2000000 	130000 	133330

'''

'''
OV12A10 12MP CMOS
    Features
    Tech Specs

1.25 µm x 1.25 µm pixel

Optical size of 1/2.8″, 6.46 mm

34.5° CRA

12MP at 30 fps

Programmable controls for:
– Frame rate
– Mirror and flip
– Cropping
– Windowing

supports images sizes:
– 12MP (4096×3072)
– 4K2K (3840×2160)
– 1080p (1920×1080), and more

416 bytes of embedded one-time programmable (OTP) memory for customer use

Support for output formats:
– 10-bit RGB RAW

Two-wire serial bus control (SCCB)

MIPI serial output interface (1-lane, 2-lane, or 4-lane)

Two on-chip phase lock loops (PLLs)

2x binning support

Image quality controls:
– Defect pixel correction
– Automatic black level calibration
– Lens shading correction

Built-in temperature sensor

Suitable for module size of 8.5 x 8.5 x less than 5 mm


'''

'''
### TABLE C ###
===== Device 0: 000573600112 ===== (Kinect Azure)
resolution width: 640
resolution height: 576
principal point x: 320.803894
principal point y: 334.772949
focal length x: 504.344574, 0.504344574mm
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


'''
THERMAL CAMERA
160x120 (thermal)
'''


import math
import pandas
import matplotlib.pyplot as plt


def calculate_maximum(focal_length=0.504344574, resolution=(640,576), chip_width=5.4, pixel_size_um=12, fov=75, camera_distance=1, desired_pixels=3, degrees=True):
    '''
    Rs is the spatial resolution (maybe either X or Y)
    FOV is the field of view dimensions (mm) in either X or Y
    Ri is the image sensor resolution; number of pixels in a row (X dimension) or column (Y dimension)
    Rf is the feature resolution (smallest feature that must be reliably resolved) in physical units (mm)
    Fp is the number of desired pixels that will span a feature of minimum size.
    Rs = Rf / Fp
    Ri = FOV / Rs


     Object dimension*(Image dimension / focal-length) = distance to object.
    
    or

    image dimension = object_size * focal_length / distance 



    HFOV = Camera_Dist * (Chip_Width/Focal_Length)

    Nyquist frequency is 2.56 but we can round to 3
    '''

    dist_list = range(1, 10)
    ppm_list = []
    ppcm = []
    pp5mm = []
    hov_list = []
    edge_resolution_list = []
    for camera_distance in range(1, 10):

        camera_distance = camera_distance/10 # We calculate pixels per 10cm

        x_resolution = resolution[0]
        HFOV = camera_distance * (chip_width/focal_length) # Meters horizontal fov at camera distance
        ppm = x_resolution / HFOV # Pixels per meter
        f_length = (camera_distance * chip_width * ppm) / x_resolution
        

        

        

        

        edge_resolution = ppm / math.cos(fov/2) # max_fov is described in table A
        
        ppm_list.append(ppm)
        hov_list.append(HFOV)
        edge_resolution_list.append(edge_resolution)


        print("Distance in meters:", camera_distance)
        print("HFOV:", HFOV)
        print("Pixels per M:", ppm)
        print("Pixels per CM:", ppm/100)
        ppcm.append(ppm/100)
        pp5mm.append(ppm/200)
        print("Pixels per 5mm:", ppm/200)
        
        print("Calculated Focal Length:", f_length, "  Manufacturer Focal Length:", focal_length)
        print("Edge Resolution:", edge_resolution)
        print()


    df = pandas.DataFrame(list(zip(dist_list, ppcm, pp5mm, edge_resolution_list)),
                columns =['Distance(m)', 'PPCM', 'PP5MM', 'Edge Resolution'])



    df.plot(x='Distance(m)', y="PPCM")
    plt.show()
    # distance = range(1, 1000) # 1mm to 10m
    # for d in distance:
    # pixels_per_distance = feature_size * (focal_length)
    # print(pixels_per_distance)
        # pixels_per_mm = list() # calculates pixels per_ 
    
    # if degrees:
    #     fov = fov * (math.pi/180)

    # # # FOV can be expressed as pixels per inch
    # # #Distance per 10m
    # #         # / |
    # #       #  /  |
    # #       # /   |
    # #       #/    | ?
    # #      #/     |
    # #     #/      |
    # # #.  /_______|
    # # # fov/2   10m
    # distance = range(1, 1000) # 1mm to 10m
    # for d in distance:
    #     fov_distance = (math.tan(fov/2)) * 2 # Size of pixels at 1mm
        # print(fov_distance)

        # print("Pixel size at", d, "cm: ", fov_distance * d, "mm")
        # print(d*fov_distance)
            # print(d)
            # break

    # print(fov_distance)

    # ppm = resolution[0] / fov_distance
    # print("Pixels/m =", ppm)


    # print()
    # # Rs = feature_resolution / desired_pixels
    # # Ri = fov_distance / Rs
    # return Ri

def exposure_value(f_stop, exposure_time):
    '''
     In a mathematical expression involving physical quantities, it is common practice to require that the argument to a transcendental function (such as the logarithm) be dimensionless. The definition of EV ignores the units in the denominator and uses only the numerical value of the exposure time in seconds; EV is not the expression of a physical law, but simply a number for encoding combinations of camera settings.
    '''
    return math.log2((f_stop**2)/exposure_time)

def digital_number(rgb_img, exposure_time_s, f_stop, iso, scene_lumin):
    h,w,d = rgb_img.shape
    for i in range(h):
        for j in range(w):
            exposure_time_s
            rgb_img[i,j] 

# def calculate_avg_lumin(rgb_image, exposure_time_s, f_stop, iso, scene_lum):



if __name__ == "__main__":
    # Depth
    # calculate_maximum()
    
    # RGB
    # calculate_maximum(resolution=(1920,1080),chip_width=6.46,fov=90)
    # calculate_maximum(resolution=(2560,1440),chip_width=6.46,fov=90)
    # calculate_maximum(resolution=(3840,2160),chip_width=6.46,fov=90)

    # FLIR one pro
    chip_width= 160*(12/1000) # = 1.92
    calculate_maximum(resolution=(160,120),chip_width=chip_width,fov=55)
    calculate_maximum(resolution=(1440,1080),chip_width=chip_width,fov=55)
    
    
