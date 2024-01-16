
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import KMeans
import sklearn.tree

from scipy.spatial import distance

import argparse
import utils

import numpy as np
import cv2

# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from pysptools import eea

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist
def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar

def get_endmembers(data, q,  mask=None, path="./EE", suffix=None ,header=None, output=True):
    print('Endmembers extraction with PPI')
    ee = eea.PPI()
    atgp = eea.ATGP()
    E = atgp.extract(data,q)
    # U = ee.extract(data, q, maxit=5, normalize=True, ATGP_init=True, mask=mask)
    U = ee.extract(data, 10)
    print(U)
    print(E)

    if output == True:
        ee.display(axes=header, suffix=suffix)
        atgp.display(axes=header, suffix=suffix)
    return np.unique(E, axis=0)

if __name__ == "__main__":
    # matplotlib.use('pdf') 
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    # ap.add_argument("-c", "--clusters", required = True, type = int,
    #     help = "# of clusters")
    # args = vars(ap.parse_args())

    # # load the image and convert it from BGR to RGB so that
    # # we can dispaly it with matplotlib
    # image = cv2.imread(args["image"])
    image = cv2.imread('C:/Users/legom/Pictures/fish.jpg')
    image = cv2.resize(image, (200,200), interpolation = cv2.INTER_AREA)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)


    print("creating image")
    img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    b,g,r = cv2.split(img_color)
    # print("creating plt")
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1, projection="3d")
    # pixel_colors = img_color.reshape((np.shape(img_color)[0]*np.shape(img_color)[1], 3))
    # norm = colors.Normalize(vmin=-1.,vmax=1.)
    # norm.autoscale(pixel_colors)
    # pixel_colors = norm(pixel_colors).tolist()
    # axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    # axis.set_xlabel("Red")
    # axis.set_ylabel("Green")
    # axis.set_zlabel("Blue")
    # print("showing image")
    

    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1, projection="3d")
    # axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    # axis.set_xlabel("Hue")
    # axis.set_ylabel("Saturation")
    # axis.set_zlabel("Value")
    # plt.show()

    # clf = sklearn.tree.DecisionTreeClassifier()
    # # clf.fit(h,s,v)
    # clf.tree_.apply(np.asfortranarray(X.astype(sklearn.tree._tree.DTYPE)))

    endmembers = get_endmembers(hsv_img, 20,output=False)
    #Calculate Euclidean distance between 3 values

    # #Cluster it
    image = hsv_img.reshape((hsv_img.shape[0] * hsv_img.shape[1], 3))
    kmeans = KMeans(n_clusters = len(endmembers), init = endmembers)
    clusters = kmeans.fit_predict(image)

    # dbscan = DBSCAN(eps=10, min_samples = 20)
    for pixel in endmembers:
        for pixel2 in endmembers:
            print(distance.euclidean(pixel, pixel2))
            if distance < 150:
                endmembers.del()
            
    
    # clusters = dbscan.fit_predict(image)

    # fig = plt.figure()
    
    # axis = fig.add_subplot(1, 1, 1, projection="3d")
    # axis.scatter(h.flatten(), s.flatten(), v.flatten(), c=clusters, cmap="viridis")
    # axis.set_xlabel("Hue")
    # axis.set_ylabel("Saturation")
    # axis.set_zlabel("Value")
    # plt.show()

    print(clusters)
    # grab the image dimensions
    h = hsv_img.shape[0]
    w = hsv_img.shape[1]

    clusters = np.reshape(clusters, (h,w))
    print(clusters.shape)
    print(hsv_img.shape)
# loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            # image[y, x] = 255 if image[y, x] >= T else 0
            hsv_img[y,x] = np.array([clusters[y,x]*50,clusters[(y,x)]*50,clusters[(y,x)]*50])
            
            # hsv_img[y,x] = clusters[(x+y)]
    print(hsv_img.shape)
    cv2.imshow("hsv_image",hsv_img)
    cv2.waitKey(1)


    
    # plt.show()
    # print(clustering.labels_)
    # labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.


    # colors = [int(i % 23) for i in labels[0]]
    # axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=colors, marker=".")
    # axis.set_xlabel("Hue")
    # axis.set_ylabel("Saturation")
    # axis.set_zlabel("Value")
    # # Black removed and is used for noise instead.
    # # unique_labels = set(labels)
    # # print(unique_labels)

    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    # plt.imshow(image)
    # cluster the pixel intensities
    # clt = KMeans(n_clusters = 5, init=endmembers)

    # clt.fit(image)
    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    # hist = centroid_histogram(clustering)
    # # print(bar)
    # bar = plot_colors(hist, clustering)
    # show our color bart
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    plt.show()


