import open3d
import cv2
import sys
import numpy as np
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QAction, QCheckBox,
                             QComboBox, QDockWidget, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSpinBox, QStackedLayout, QStyle,
                             QTextEdit, QVBoxLayout, QWidget, QMenuBar, QGraphicsView, QToolButton, QGraphicsScene, QGraphicsPixmapItem, QFrame)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import QEvent
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QBrush, QPen

from copy import deepcopy
from plyfile import PlyData, PlyElement
import os
import utils

class App(QWidget):

    def __init__(self):
        super().__init__()
        # self.graphicsView.setMouseTracking(True)
        self.setMouseTracking(True)
        self.title = 'Label Image'
        self.left = 50
        self.top = 50
        self.width = 640
        self.height = 480
        self.zoom_height = 480

        self.file_list = []

        
        

        self.edit_mode = False
        self.mouse_pos = (0,0)

        self.hover_point = None
        self.selected_point = None
         
        self.current_polygon = None

        self.dataset = Dataset()
    
        self.current_folder, self.file_list = self.open_ply_folder()

        self.init_buttons()
        self.initUI()
        

        self.file_list_widget = QListWidget()

        # self.file_list_widget.setSelectionMode(
        #     QAbstractItemView.ExtendedSelection
        # )
    def load_image():
        pass

    @QtCore.pyqtSlot()
    def show_image(self, image):
        image = np.nan_to_num(image).astype(np.uint8)
        # self.setGeometry(self.left, self.top, image.shape[1], image.shape[0])
        self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.q_image = QPixmap.fromImage(self.q_image)
        # self.zoom_height = self.q_image.height()
        # self.image_frame.setPixmap(self.q_image)
        self.viewer.setPhoto(self.image, self.q_image)
        
    

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.viewer = PhotoViewer(self, self.dataset)
        self.layout.addWidget(self.viewer)



        # self.layout.setAlignment(Qt.AlignTop)
        self.header = QHBoxLayout()
        self.body = QVBoxLayout()

        self.layout.addLayout(self.header)
        self.layout.addLayout(self.body)

        self.init_menu()
        self.init_image_UI()

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # self.show_image(self.image)
        self.viewer.setPhoto(self.image)
        # self.viewer.fitInView()

        self.add_label()
        self.show()


    def init_image_UI(self):
        # try:
        print(self.file_list[0][:-3])
        if self.file_list[0][-3:] == "ply":
            print("Plyfile found")
            self.dataset.load_ply(self.get_global_filename(self.current_folder, self.file_list[0]))
            # self.read_ply(self.get_global_filename(self.current_folder, self.file_list[0]))
            self.image = self.dataset.image
        else:
            self.image = cv2.imread(self.get_global_filename(self.current_folder, self.file_list[0]))
        
        # print(self.image)
        # print(self.image[0])
        # print(self.image[0][0])
        # print(self.image.shape)
        self.original_img = deepcopy(self.image)
        # self.scene = QGraphicsScene(self)
        # self.graphics = QGraphicsPixmapItem(self.q_image)
        # self.scene.addItem(self.graphics)
        # self.graphicsView.setScene(self.scene)
        # self.scale = 1
        
        # self.image_frame = QLabel()
        # self.image_frame.setMouseTracking(True)
        # self.layout.addWidget(self.image_frame)


    def init_file_UI(self):
        pass
        
        # # Create widget
        # self.image = QLabel(self)
        # pixmap = QPixmap('ras/CreateDataset/Screenshot 2021-01-03 172810.jpeg')
        # self.image.setPixmap(pixmap)
        # self.resize(pixmap.width(),pixmap.height())
        
        # self.show()

    def init_buttons(self):
        self.drag_start = False
    
    def init_menu(self):
        bar = QMenuBar()
        bar.setMaximumHeight(20)
        
        self.header.addWidget(bar)
        
        file = bar.addMenu("File")
        edit = bar.addMenu("Edit")

        edit_toggle = QAction("Toggle Edit Mode", self)
        edit_toggle.triggered.connect(lambda: self.viewer.toggle_edit())

        edit.addAction(edit_toggle)

    # def on_zoom_in(self, event):
    #     self.zoom_height += 100
    #     self.resize_image()

    # def on_zoom_out(self, event):
    #     self.zoom_height -= 100
    #     self.resize_image()

    # def resize_image(self):
    #     image = np.nan_to_num(self.image).astype(np.uint8)
    #     self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
    #     self.q_image = QPixmap.fromImage(self.q_image)
    #     # self.zoom_height = self.q_image.height()

    #     self.q_image = self.q_image.scaledToHeight(self.zoom_height)
    #     self.image_frame.setPixmap(self.q_image)
    #     # self.image_frame.setPixmap(self.q_image)
        


    # def mousePressEvent(self, event):
    #     self.drag_start = True
    #     print("drag start")
    #     mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))
    #     # self.map_to_widget(self.image_frame, point)

    #     if self.current_polygon is not None:
            
    #         if event.button() == Qt.LeftButton and not self.edit_mode:
    #             print("Left Button Clicked")
    #             self.current_polygon.assign_point(mapped_point)
    #             image = self.current_polygon.draw(self.image)
    #             self.show_image(image)


    #         if event.button() == Qt.RightButton:
    #             print("Right button clicked")
    #             self.current_polygon.end_poly()
    #             image = self.current_polygon.draw(self.image)
    #             mask = self.current_polygon.create_mask(self.image.shape[0], self.image.shape[1])

    #             cropped , coordinates, cloud = self.current_polygon.get_segment_crop(self.original_img, self.dataset.point_pairs)
    #             self.show_image(image)
            
    #         if event.buttons() == QtCore.Qt.LeftButton and self.edit_mode:
                
    #             point = self.current_polygon.check_near_point(mapped_point, dist=5)
    #             self.selected_point = point
                

    #             # self.selected_point = poin

    #             # self.current_polygon = None


    # def mouseReleaseEvent(self, event):
    #     mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))
    #     self.drag_start = False
    #     print("dropped")

    #     if self.selected_point is not None and self.edit_mode:
    #         self.redraw_image()
    #         self.current_polygon.edit_point(self.selected_point, mapped_point)
    #         self.current_polygon.draw(self.image)
            
    #     self.selected_point = None


    # def mouseMoveEvent(self, event):
    #     self.mouse_pos = (event.x(), event.y())
    #     self.repaint()
        
    #     mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))

    #     if event.buttons() == QtCore.Qt.NoButton:
    #         if self.current_polygon is not None:
    #             point = self.current_polygon.check_near_point(mapped_point, dist=5)

    #             if point is not None:
                    
    #                 image = cv2.circle(self.image, point, 1, (0,255,0), 5)
    #                 # image = cv2.circle(self.image, point, 5, (0,0,0), 1)
    #                 self.hover_point = point
    #                 self.show_image(image)

    #             elif self.hover_point is not None and point is None:
                    
    #                 image = cv2.circle(self.image, self.hover_point, 1, (0,0,255), 4)
    #                 image = cv2.circle(self.image, self.hover_point, 4, (0,0,0), 2)
    #                 self.hover_point = None

    #                 self.show_image(image)

    #     if self.drag_start:
    #         # print("dragging")
    #         if self.current_polygon is not None:
    #             point = self.current_polygon.check_near_point(mapped_point,dist=12)

    #             if point is None and not self.edit_mode:
    #                 self.current_polygon.assign_point(mapped_point)
    #                 image = self.current_polygon.draw(self.image)
    #                 self.show_image(image)

    # def drawPoints(self,qp):

    #       qp.setPen(Qt.red)
    #       size = self.size()
    #       x=self.x
    #       y=self.y
    #       qp.drawPoint(x,y)   

    # def paintEvent(self, event):
    #     qp = QPainter(self.q_image)
    #     qp.setRenderHint(QPainter.Antialiasing)
    #     pen = QPen(Qt.red, 7)
    #     brush = QBrush(Qt.red)
    #     qp.setPen(pen)
    #     qp.setBrush(brush)
    #     point = QtCore.QPoint(self.mouse_pos[0], self.mouse_pos[1])
    #     qp.drawPoint(point)
    #         # for i in range(self.points.count()):
    #         #     qp.drawEllipse(self.points.point(i), 5, 5)
    #         # or 
    #         # qp.drawPoints(self.points)
    # def redraw_image(self):
    #     self.image = deepcopy(self.original_img)
        
    # def dragEnterEvent(self, event):
    #     event.acceptProposedAction()
    #     if event.source() is not self:
    #         self.clicked.emit()
    
    def add_label(self):
        print("Creating label")
        self.dataset.create_polygon(self.get_global_filename(self.current_folder, self.file_list[0]))
        # self.current_polygon = self.dataset.current_polygon
        self.viewer.set_polygon(self.dataset.current_polygon)

    def map_to_widget(self, widget, point):
        # gp = widget.mapToGlobal(QtCore.QPoint(0, 0))
        # b_a = widget.mapFromGlobal(gp)
        # print()
        # # print(b_a.x(),b_a.y())
        mapped = ((point[0] - widget.pos().x()), point[1] - widget.pos().y())

        return mapped

    def toggle_edit(self):
        self.edit_mode = not self.edit_mode

    def open_ply(self):
        file = str(QFileDialog.getOpenFileName(self, "Select Directory")[0])


    def open_ply_folder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        file_list = os.listdir(folder)
        print(folder,file_list)
        return folder, file_list
    
    def get_global_filename(self, folder, filename):
        print(str((folder + "/" + filename)))
        return str((folder + "/" + filename))

    def read_ply(self, filename):
        plydata = PlyData.read(filename)
        image, depth, point_pairs = project_2D(REALSENSE_D415_INTRINSICS, plydata)
        cloud = map_pairs_2D(image, point_pairs)
     
        
        # self.label.setText('Mouse coords: ( %d : %d )' % (event.x(), event.y()))
class Dataset():
    def __init__(self):
        # self.labels = []
        # self.polygons = []

        # labels[filename] = (polygons, labels)
        self.labels = {}
        self.current_polygon = None
        self.image = None
        self.depth = None
        self.point_pairs = None
        self.cloud = None

    def create_polygon(self, filename):
        self.current_polygon = LabelPolygon()
        self.current_polygon.creating = True
        self.current_polygon.filename = filename
        
        # self.polygons.append(LabelPolygon)

        if filename in self.labels:
            #grab the already created list, and append a new polygon
            new_label = self.labels[filename].value()
            new_label.append(self.current_polygon)
        else:
            #new label encapsulated with list so it can append additional polygons later
            new_label = [self.current_polygon]
            self.labels[filename] = new_label

        # print(self.labels)

    # def add_label(self):

    def get_polygon(self, int):
        pass

    def load_ply(self, filename):
        plydata = PlyData.read(filename)
        self.image, self.depth, self.point_pairs = utils.project_2D(utils.REALSENSE_D415_INTRINSICS, plydata)
        self.cloud = utils.map_pairs_2D(self.image, self.point_pairs)

class LabelPolygon():
    def __init__(self):
        self.polygon_label = None
        self.filename = None
        self.points = []

        # flag to see if polygon is being created
        self.creating = False 

        #Red in BGR
        self.poly_colour = (0,0,255)

    def assign_point(self, location):
        """
        Creates a point in the polygon
        """
        self.points.append(location)
        
    def start_poly(self):
        """
        Starts a new polygon (May not use?)
        """
        self.creating = True

    def end_poly(self):
        """
        Ends the polygon closing it up
        """
        self.creating = False

    def draw(self, image):
        """
        Draws the polygon with points and edges with an infill
        """
        overlay = image.copy()
        overlay = np.nan_to_num(overlay).astype(np.uint8)
        # print(overlay)
        output = image.copy()
        output = np.nan_to_num(overlay).astype(np.uint8)



            
        for point in self.points:
            # image = cv2.circle(overlay, point, 1, (0,10,255),5)
            # image = cv2.circle(overlay, point, 4, (0,0,0), 2)
            image = cv2.circle(output, point, 1, (0,150,255), 1)

        # Draw Lines (Yellow)
        if len(self.points) >= 2:
            pts = np.array(self.points, np.int32) 
  
            pts = pts.reshape((-1, 1, 2)) 

            if not self.creating:

                image = cv2.fillPoly(overlay, [pts], (0,0,255,0.5))

            image = cv2.polylines(overlay,[pts], (not self.creating), (0,255,255),1 )

        cv2.addWeighted(overlay, 0.3, output, 1 - 0.3,
		0, output)
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

    def get_point(self, clicked_location):
        """
        Returns point if clicked
        """
        pass

    def check_near_point(self, location, dist=10):
        for point in self.points:
            if (location[0]-point[0])**2 + (location[1]-point[1])**2 <= dist**2:
                return point
        return None

    def check_in_polygon(self, location):
        pass
    
    def create_mask(self,width, height):
        pts = np.array(self.points, np.int32) 
        pts = pts.reshape((-1, 1, 2)) 
        black_image = np.zeros((width, height), np.uint8)
        mask = cv2.fillPoly(black_image, [pts], (255,255,255))
        # cv2.imshow("Mask", mask)
        return mask
                # image = cv2.circle(image, point, 1, (0,255,0),7)
    
    def get_segment_crop(self, image,point_pairs):
        
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
        dst2 = bg+ dst
        full_dst2 = full_bg + full_dst 
        full_invalid_dst2 = full_bg_invalid + full_dst

        cloud = utils.map_pairs_2D(full_invalid_dst2, point_pairs)
        # utils.display_cloud(cloud)
        utils.edit_cloud(cloud)
        

        # np.nan_to_num(full_dst2).astype(np.uint8)
        # cv2.imshow("FullDst", full_dst2)
        # cv2.imshow("Cropped Image", dst2)
        
        # cv2.waitKey(0)
        return dst2, coordinates, cloud

class PhotoViewer(QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent, dataset):
        super(PhotoViewer, self).__init__(parent)
        self.parent = parent
        self.image = None
        self.original_img = None
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self._painter = QtGui.QPainter()
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

        self.hover_point = None
        self.drag_start = False
        self.edit_mode = False
        self.mouse_pos = None
        self.selected_point = None
        self.setDragMode(QGraphicsView.NoDrag)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    def update_image(self, image):
        image = np.nan_to_num(image).astype(np.uint8)
        # self.setGeometry(self.left, self.top, image.shape[1], image.shape[0])
        self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.q_image = QPixmap.fromImage(self.q_image)
        # self.zoom_height = self.q_image.height()
        # self.image_frame.setPixmap(self.q_image)
        self.setPhoto(image, self.q_image)

    def toggle_edit(self):
        self.edit_mode = not self.edit_mode

    def set_polygon(self, current_polygon):
        self.current_polygon = current_polygon

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def redraw_image(self):
        self.image = deepcopy(self.original_img)

    def setPhoto(self, image, pixmap=None):
        if self.image is None:
            self.image = image
            self.original_img = deepcopy(self.image)
        else:
            self.image = image
        # print(image)
        image = np.nan_to_num(image).astype(np.uint8)
        pixmap = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(pixmap)

        # self._zoom = 100
        if pixmap and not pixmap.isNull():
            self._empty = False
            # self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
            # print(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

        self.drag_start = True
        print("drag start")
        # mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))
        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())
        # self.map_to_widget(self.image_frame, point)

        if self.current_polygon is not None:
            
            if event.button() == Qt.LeftButton and not self.edit_mode:
                print("Left Button Clicked")
                self.current_polygon.assign_point(mapped_point)
                self.redraw_image()
                image = self.current_polygon.draw(self.image)
                self.update_image(image)


            if event.button() == Qt.RightButton:
                print("Right button clicked")
                self.current_polygon.end_poly()
                self.redraw_image()
                image = self.current_polygon.draw(self.image)
                mask = self.current_polygon.create_mask(self.image.shape[0], self.image.shape[1])

                cropped , coordinates, cloud = self.current_polygon.get_segment_crop(self.original_img, self.parent.dataset.point_pairs)
                self.update_image(image)
            
            #Edit mode select point
            if event.buttons() == QtCore.Qt.LeftButton and self.edit_mode:
                #Distance for selected point (should match mouse move event and change colour to indicate accurate selection)
                point = self.current_polygon.check_near_point(mapped_point, dist=2)
                self.selected_point = point
                

                # self.selected_point = poin

                # self.current_polygon = None
                
    # def paintEvent(self, event):
    # #     print("Paint event")
    #     if self.mouse_pos is not None:
    #         print("painting")
    #         self.draw_points()


    def draw_points(self):
        if self.drag_start:
            self._painter.begin(self._scene)
            
            self._painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(Qt.red, 7)
            brush = QBrush(Qt.red)
            self._painter.setPen(pen)
            self._painter.setBrush(brush)
            point = QtCore.QPoint(self.mouse_pos[0], self.mouse_pos[1])
            self._painter.drawPoint(point)
            # self._painter.drawPixmap(point)
            self._painter.end()
        
    def mouseReleaseEvent(self, event):
        # mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))
        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())

        self.drag_start = False
        print("dropped")
        

        if self.selected_point is not None and self.edit_mode:
            self.redraw_image()
            self.current_polygon.edit_point(self.selected_point, mapped_point)
            
        image = self.current_polygon.draw(self.image)
        self.update_image(image)
            
        self.selected_point = None


    def mouseMoveEvent(self, event):
        self.mouse_pos = (event.x(), event.y())
        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())
        # self.mouse_pos = (event.x(), event.y())
        # self.repaint()
        
        # mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))

        try:
            image = np.nan_to_num(self.original_img).astype(np.uint8)
            b,g,r = image[mapped_point[1]][mapped_point[0]]

            self.redraw_image()
            image = cv2.circle(image, mapped_point, 1, (int(255-b),int(255-g),int(255-r)))
            # image = cv2.rectangle(self.image, mapped_point, mapped_point, (255-b,255-g,255-r))
            image = self.current_polygon.draw(image)
            # image = cv2.circle(self.image, self.hover_point, 1, (0,255,255), 1)
            self.update_image(image)
        except:
            print("Could not grab mapped point.")

        if event.buttons() == QtCore.Qt.NoButton:

            if self.current_polygon is not None:
                #Check when hovering is within distance value to change colour
                point = self.current_polygon.check_near_point(mapped_point, dist=2)

                #When hovering over point
                if point is not None:
                    print("GReen?")
                    image = cv2.circle(self.image, point, 1, (0,255,0), 1)
                    # image = cv2.circle(self.image, point, 5, (0,0,0), 1)
                    self.hover_point = point
                    self.update_image(image)

                #When hover point is left alone
                elif self.hover_point is not None and point is None:
                    # self.redraw_image()
                    
                    image = cv2.circle(self.image, self.hover_point, 1, (0,255,255), 1)
                    # image = cv2.circle(self.image, self.hover_point, 4, (0,0,0), 2)
                    self.hover_point = None

                    self.update_image(image)

        if self.drag_start:

            # self.draw_points()
            
            print("dragging")
            if self.current_polygon is not None:
                #Check if hover is within distance for assigning a point (distance between new points, used for hold and drag assignment)
                point = self.current_polygon.check_near_point(mapped_point,dist=5)

                if point is None and not self.edit_mode:
                    self.redraw_image()
                    self.current_polygon.assign_point(mapped_point)
                    image = self.current_polygon.draw(self.image)
                    self.update_image(image)

        # else:
            
            # image = cv2.circle(self.image, self.hover_point, 4, (0,0,0), 2)
            # self.hover_point = None


def cloud_to_image():
    pass

def image_to_cloud():
    pass

def display_2D():
    pass

def display_3d():
    pass

def create_polygon():
    pass

def label_polygon():
    pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())