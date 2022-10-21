import random
from datetime import datetime
import numpy as np
import cv2
import labeling.utils as utils


'''
######################################################################
CLASS LABELPOLYGON


######################################################################
'''
class LabelPolygon():
    def __init__(self):
        random.seed(datetime.now())
        self.mean = (0,0)
        self.polygon_label = None
        self.filename = None
        self.points = []
        self.view = True
        self.index = None
        self.color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255),0.5) #RGBA

        self.queue_save = False

        # flag to see if polygon is being created
        self.creating = True

    def set_label(self, label):
        self.polygon_label = str(label)

    def calculate_mean(self):
        '''
        Calculates the mean of all the points
        '''
        mean_x = 0
        mean_y = 0

        for point in self.points:
            mean_x += point[0]
            mean_y += point[1]

        self.mean = (mean_x/len(self.points), mean_y/len(self.points))

    # def calculate_median(self):
    #     '''
    #     Calculates the mean of all the points
    #     '''
    #     mean_x = 0
    #     mean_y = 0

    #     for point in self.points:
    #         mean_x += point[0]
    #         mean_y += point[1]

    #     self.mean = (mean_x/len(self.points), mean_y/len(self.points))

    
    def assign_point(self, location):
        """
        Creates a point in the polygon
        """
        # self.queue_save = True
        self.points.append(location)
        self.calculate_mean()

    # def get_points(self, point_idx):
    #     '''
    #     '''

    def start_poly(self):
        """
        Starts a new polygon (May not use?)
        """
        self.queue_save = True
        self.creating = True

    def end_poly(self):
        """
        Ends the polygon closing it up self.polygon
        """
        self.queue_save = True
        self.creating = False

    def draw(self, image, infill=(0,0,255,0.5), line_color=(0,255,255), thickness=1, overwrite_label=None, show_label=True, show_points = True, show_lines = True):
        """
        Draws the polygon with points and edges with an infill
        """
        overlay = image.copy()
        overlay = np.nan_to_num(overlay).astype(np.uint8)
        # print(overlay)
        output = image.copy()
        output = np.nan_to_num(overlay).astype(np.uint8)

        if show_points:
            for point in self.points:
                # image = cv2.circle(overlay, point, 1, (0,10,255),5)
                # image = cv2.circle(overlay, point, 4, (0,0,0), 2)
                image = cv2.circle(output, point, 1, (0,150,255), 1)

        # Draw Lines (Yellow)
        
        if len(self.points) >= 2:
            pts = np.array(self.points, np.int32) 
            pts = pts.reshape((-1, 1, 2)) 

            if not self.creating:
                image = cv2.fillPoly(overlay, [pts], self.color)

            if show_lines:
                image = cv2.polylines(overlay,[pts], (not self.creating), line_color, thickness)
            
            if show_label:
                font_color = (255, 245, 85)
                label = self.polygon_label
                if overwrite_label is not None:
                    label = overwrite_label
                    
                if label:
                    cv2.putText(output, text=label, org=(int(self.mean[0]) - 20, int(self.mean[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=font_color,thickness=1, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(output, text="No Label", org=(int(self.mean[0]) - 20, int(self.mean[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=font_color,thickness=1, lineType=cv2.LINE_AA)

            
        cv2.addWeighted(overlay, 0.3, output, 1 - 0.3, 0, output)

        return output
    
    # def draw_qt()

    def edit_point(self, point_start, point_stop):
        """
        Moves a point when clicked and remains clicked
        """
        for i, point in enumerate(self.points):
            if point == point_start:
                self.points[i] = point_stop
                print("Replaced", point, " to ", point_stop)
        # pass

    def remove_point(self, point):
        """
        Removes a point from the polygon
        """
        self.queue_save = True
        for i, test_point in enumerate(self.points):
            if point == test_point:
                self.points.pop(i)
                

        

        

    def get_point(self, clicked_location):
        """
        Returns point if clicked
        """
        pass
    
    def get_adjacent(self, point):
        """
        Returns the two adjacent points, if a point is unavailable, that part of the tuple returns None

            Returns (Previous, Next)
            if Previous is none: Returns (None, Next)
            if Next is none: Returns (Previous, None)
            if No adjacent points: Returns (None, None)
        """
        adjacent = (None, None)
        point_index = self.points.index(point)
        #if not beginning or end, has 2 adjacent points
        if len(self.points) > point_index + 1 and point_index >= 1:
            adjacent = (self.points[point_index-1], self.points[point_index+1])
        # Point is at the end, so grab prior point
        elif len(self.points) == point_index + 1:
            adjacent = (self.points[point_index-1], None)
        elif point_index == 0:
            adjacent = (None, self.points[point_index+1])

        return adjacent

    def check_near_point(self, location, dist=10):
        for point in self.points:
            if (location[0]-point[0])**2 + (location[1]-point[1])**2 <= dist**2:
                return point
        return None

    def check_in_polygon(self, location):
        pass
    
    def create_mask(self, width, height):
        pts = np.array(self.points, np.int32) 
        pts = pts.reshape((-1, 1, 2)) 
        black_image = np.zeros((width, height), np.uint8)
        mask = cv2.fillPoly(black_image, [pts], (255,255,255)) 
        # cv2.imshow("Mask", mask)
        return mask
                # image = cv2.circle(image, point, 1, (0,255,0),7)

    def get_segment_crop(self, image, point_pairs):
        
        # print(image)
        # print(point_pairs)

        pts = np.array(self.points)

        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        image = np.nan_to_num(image).astype(np.uint8)
        croped = image[y:y+h, x:x+w].copy()

        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        # full_mask = np.zeros(image.shape[:2], np.uint8)
        full_mask = self.create_mask(image.shape[0], image.shape[1])

        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # cv2.drawContours(full_mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        indices = np.where(mask == [255])
        coordinates = zip(indices[0], indices[1])
        # for coord in coordinates:
        #     print(coord[0], coord[1])
        # cv2.imshow("MASK!", full_mask)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        full_dst = cv2.bitwise_and(image, image, mask=full_mask)

        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255

        full_bg = np.ones_like(image, np.uint8)*255
        full_bg_invalid = np.ones_like(image, np.uint8)*-1
        # print(full_bg_invalid)
        # cv2.imshow("fullBG", full_bg)
        
        cv2.bitwise_not(bg,bg, mask=mask)
        cv2.bitwise_not(full_bg,full_bg, mask=full_mask)
        cv2.bitwise_not(full_bg_invalid,full_bg_invalid,full_mask)
        
        # full_bg_invalid[full_bg_invalid >= 255] = -1
        # full_bg_invalid = full_bg_invalid.astype(np.int16) * -1
        # print(full_bg_invalid)
        dst2 = bg + dst
        full_dst2 = full_bg + full_dst 
        full_invalid_dst2 = full_bg_invalid + full_dst

        cloud, points, colour = utils.map_pairs_2D(full_invalid_dst2, point_pairs)
        # utils.display_cloud(cloud)
        utils.edit_cloud(cloud)
        

        # np.nan_to_num(full_dst2).astype(np.uint8)
        # cv2.imshow("FullDst", full_dst2)
        # cv2.imshow("Cropped Image", dst2)
        
        # cv2.waitKey(0)
        return dst2, coordinates, cloud
