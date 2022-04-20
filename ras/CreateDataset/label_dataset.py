from datetime import datetime

# import open3d
import cv2
import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QAbstractItemView, QApplication, QAction, QCheckBox,
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
from PyQt5.QtCore import QThread, QObject, pyqtSignal

from copy import deepcopy
from plyfile import PlyData, PlyElement
import os
# from ras.CreateDataset.utils import o3d_add_object
import utils

import random

from viewer import Action_Poll

from multiprocessing import Process
# import pandas as pd
'''
######################################################################
CLASS APP


######################################################################
'''
#class App(QWidget):
class App(QMainWindow):
    stop_signal = pyqtSignal()  # make a stop signal to communicate with the worker in another thread
    def __init__(self):
        super().__init__()
        self.image = None
        # self.graphicsView.setMouseTracking(True)
        self.setMouseTracking(True)
        self.title = 'Label Image'
        self.left = 50
        self.top = 50
        self.width = 1280
        self.height = 720
        self.zoom_height = 480

        self.file_list = []
        self.selected_file = ""

        self.edit_mode = False
        self.mouse_pos = (0,0)

        self.hover_point = None
        self.selected_point = None
         
        self.current_polygon = None

        self.dataset = Dataset()
        self.label_layouts = []

        self.o3d_vis = None
        self.threads = []

        self.init_buttons()
        self.initUI()

        

        
        
        # self.file_list_widget.setSelectionMode(
        #     QAbstractItemView.ExtendedSelection
        # )
    def load_image():
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.o3d_vis is not None:
                self.stop_thread()

    @QtCore.pyqtSlot()
    def show_image(self, image):
        image = np.nan_to_num(image).astype(np.uint8)
        # self.setGeometry(self.left, self.top, image.shape[1], image.shape[0])
        #self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.q_image = QPixmap.fromImage(self.q_image)
        # self.zoom_height = self.q_image.height()
        # self.image_frame.setPixmap(self.q_image)
        self.viewer.setPhoto(self.image, self.q_image)
        
    def create_thread(self, visualizer):
        self.o3d_vis = visualizer
        thread = QThread()
        worker = Action_Poll(parent=self)
        worker.moveToThread(thread)
        thread.started.connect(lambda: worker.poll_actions(self.o3d_vis))
        # worker.updatedBalance.connect(self.updateBalance)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        return thread
    
    def start_threads(self):
        self.threads.clear()
        shapes = []
        for polygon in self.dataset.labels:
            if polygon.view:
                points_3d = utils.get_3d_from_pairs(polygon.points, self.dataset.point_pairs)
                shapes.extend(utils.o3d_polygon(points_3d))

        utils.display_cloud(self.dataset.cloud, shapes)

        self.threads = [
            self.create_thread(utils.display_cloud(self.dataset.cloud, shapes))
        ]
        for thread in self.threads:
            thread.start()

    # def init_viewerthread(self, visualizer):

    #     print("Starting Thread")
    #     # self.vis = o3d.visualization.Visualizer()
    #     # self.vis.create_window()
    #     self.worker = Action_Poll(visualizer, parent=self)
    #     # Thread:
        
    #     self.stop_signal.connect(self.worker.stop)  # connect stop signal to worker stop method
    #     self.worker.moveToThread(self.thread)
    #     # self.worker.continue_run = True
    #     # self.worker.finished.connect(self.thread.quit)  # connect the workers finished signal to stop thread
    #     # self.worker.finished.connect(self.worker.deleteLater)  # connect the workers finished signal to clean up worker
    #     # self.thread.finished.connect(self.thread.deleteLater)  # connect threads finished signal to clean up thread

    #     self.thread.started.connect(self.worker.do_work)
    #     # self.thread.finished.connect(self.worker.stop)
    #     self.thread.start()

    def new_visualizer(self):
        self.o3d_viewer.clear_geometries()

    def stop_thread(self):
        self.stop_signal.emit()  # emit the finished signal on stop
        # self.thread.stop()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)


        self.viewer = PhotoViewer(self, self.dataset)
        self.setCentralWidget(self.viewer)

        # self.layout.setAlignment(Qt.AlignTop)
        self.header = QHBoxLayout()
        self.body = QVBoxLayout()
        self.body_horizontal = QHBoxLayout()
        self.body.addLayout(self.body_horizontal)
        self.right_body = QVBoxLayout()

        self.layout.addLayout(self.header)
        self.layout.addLayout(self.body)
        
        
        #self.body_horizontal.addWidget(self.viewer)

        self.dock = QDockWidget("File List")
        self.dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        

    
        self.body_horizontal.addLayout(self.right_body)
        # fileListLayout = QVBoxLayout()
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemDoubleClicked.connect(self.open_ply)
        self.dock.setWidget(self.file_list_widget)


        self.label_list_dock = QDockWidget("Label List")
        self.label_list_widget = QListWidget()

        # self.label_list_widget.itemDoubleClicked.connect(self.open_ply) #To set the label Info Widget
        self.label_list_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.label_list_dock.setWidget(self.label_list_widget)
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_list_dock)

        #LABEL INFO
        self.label_info = QDockWidget("Label Info")
        self.label_info.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        # self.label_container_widget = QWidget()
        self.label_stack_widget = QWidget()
        self.label_stack_layout = QStackedLayout()
        self.label_stack_widget.setLayout(self.label_stack_layout)

        self.label_list_widget.clicked.connect(lambda: self.label_clicked())
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_info)

        # self.label_info_layout = QVBoxLayout()
        # self.label_container_widget.setLayout(self.label_info_layout)
        self.label_info.setWidget(self.label_stack_widget)


        self.init_menu()
        self.init_image_UI()



        # self.show_image(self.image)
        # self.add_label()
        self.show()
    
    def hide_labels(self):
        '''
        Hides labels on label_list_widget by filenames that exist in dataset
        '''
        for index in range(self.label_list_widget.count()):
            # label = self.dataset.labels[index]
            active_labels = self.dataset.file_labels[self.selected_file]

            print(index+1, active_labels)

            if index+1 not in active_labels:
                self.label_list_widget.item(index).setHidden(True)
            else:
                self.label_list_widget.item(index).setHidden(False)

            # self.label_stack_layout.setCurrentIndex(index)
            # self.current_polygon = self.dataset.labels[index]
        

    def label_clicked(self):
        self.hide_labels()
        index = self.label_list_widget.currentRow()
        self.label_stack_layout.setCurrentIndex(index)
        self.current_polygon = self.dataset.labels[index]
        self.dataset.current_polygon = self.current_polygon
        self.viewer.current_polygon = self.current_polygon
        self.redraw()

    def redraw(self):
        '''
        Resets to an original image and draws polygons, then updates the image
        '''
        self.viewer.redraw_image()
        image = self.viewer.image
        image = self.viewer.draw_polygons(image)
        self.viewer.update_image(image)
        

    def create_new_label(self):
        #self.label_stack_layout
        self.hide_labels()

        label_info_layout = QVBoxLayout()
        holder_widget = QWidget() # Add this widget dynamically
        holder_widget.setLayout(label_info_layout)

        self.label_stack_layout.addWidget(holder_widget)
        count = self.label_stack_layout.count() -1
        print(count)
        self.label_stack_layout.setCurrentIndex(count+1)
        # self.label_stack_layout.setCurrentIndex(count)

        label_name = QLineEdit()
        label_name.setPlaceholderText("Label Name")

        label_id = QLineEdit()
        label_id.setPlaceholderText("Label ID")

        distance_label = QLabel()
        distance_label.setText("Distance: 0")

        total_distance_label = QLabel()
        total_distance_label.setText("Distance: 0")

        visible = QCheckBox("Visible")
        visible.setChecked(True)
        # visible.setEnabled(False)
        
        parent_label = QComboBox()
        delete = QPushButton("Delete")
        # delete.clicked.connect(lambda: )
        preview = QPushButton("Preview")

        label_info_layout.addWidget(label_name)
        label_info_layout.addWidget(label_id)
        label_info_layout.addWidget(distance_label)
        label_info_layout.addWidget(total_distance_label)
        label_info_layout.addWidget(visible)
        label_info_layout.addWidget(parent_label)
        label_info_layout.addWidget(delete)
        label_info_layout.addWidget(preview)

        item = QListWidgetItem()
        item.setText(("Label_" + str(count)))
        item.setCheckState(2)
        print(count)
    

        self.label_list_widget.addItem(item)
        self.label_list_widget.setCurrentRow(count + 1)
        # self.label_list_widget.item
        # self.label_list_widget.itemSelectionChanged()
        # self.label_list_widget.itemChanged.connect(lambda: self.change_visible(count-1, visible_box=visible))
        
        print(label_name.text())
        label_name.textChanged.connect(lambda: self.handle_label_name_change(label_name, count))
        visible.clicked.connect(lambda: self.change_visible(count, item.checkState(), visible))
        preview.clicked.connect(lambda: self.show_labels_3D())

    def get_current_info_widgets(self, item_at):
        # layout = self.label_stack_layout.children()[0].children()[0].children()
        holder = self.label_stack_layout.itemAt(self.label_stack_layout.currentIndex()).widget() # We get Holder Widget
        layout = holder.layout()
        return layout.itemAt(item_at).widget()
    
    def get_info_widget(self, index ,item_at):
        holder = self.label_stack_layout.itemAt(index).widget() # We get Holder Widget
        layout = holder.layout()
        return layout.itemAt(item_at).widget()

    def select_label(self):
        index = self.parent_label.currentIndex()
        
    def handle_label_name_change(self, name_widget, label_index):
        print("SELECTED")
        new_text = name_widget.text()
        self.dataset.current_polygon.set_label(new_text)
        self.label_list_widget.item(label_index).setText(new_text)
        self.redraw()

    def hide_all(self):
        for index in range(self.label_list_widget.count()):
            self.get_info_widget(index, 4).setCheckState(False)
            
        for label in self.dataset.labels:
            label.view = False
        
    def show_selected_file(self):
        label_indexes = self.dataset.file_labels[self.selected_file]
        print("LABELS", self.selected_file)
        for index in label_indexes:
            label = self.dataset.get_polygon(index-1)
            label.view = True
        # self.file_list_widget.selectedItems()[0].text()


    def change_visible(self, index, set_value=None, visible_box=None):
        # print(set_value)
        # checkbox = self.label_list_widget.item(index)
        # if set_value is not None:
        #     print(self.dataset.labels[index])
        #     self.dataset.labels[index].view = set_value
        #     checkbox.setCheckState(set_value)
        #     if visible_box:
        #         visible_box.setChecked(set_value)
            # We need to set this value to 2 because 1 is partially checked (Square box instead of checkmark)
            # if set_value is True:
            #     checkbox.setCheckState(True)
            # if set_value is False:
            #     checkbox.setCheckState(False)
        # else:
        #     if not self.dataset.labels[index].view is True:
        #         checkbox.setCheckState(2)
        #     else:
        #         checkbox.setCheckState(0) 
        # if value == False:
        #     print("Setting to true")
        #     self.dataset.labels[index].view = False
        #     if visible_box:
        #         visible_box.setChecked(False)
        #     checkbox.setCheckState(2)
            
        # else:
        #     self.dataset.labels[index].view = True
        #     checkbox.setCheckState(True)
        #     if visible_box:
        #         visible_box.setChecked(True)
        
        

        # self.dataset.current_polygon.view = not self.dataset.current_polygon.view
        self.dataset.labels[index].view = not self.dataset.labels[index].view
        # checkbox.setSelected(self.dataset.labels[index].view)
        # self.label_stack_layout.setCurrentIndex(index)
        # self.label_stack_widget
        self.redraw()
        
    def show_labels_3D(self, index=None):
        '''
        '''
        if self.o3d_vis is None:
            self.start_threads()
        # if self.o3d_vis is not None:
        #     self.o3d_vis.clear_geometries()

        # shapes = []
        # # self.dataset.labels[index]
        # for polygon in self.dataset.labels:
        #     if polygon.view:
        #         points_3d = utils.get_3d_from_pairs(polygon.points, self.dataset.point_pairs)
        #         shapes.extend(utils.o3d_polygon(points_3d))
        # self.o3d_vis = utils.display_cloud(self.dataset.cloud, shapes)
        # self.init_viewerthread(self.o3d_vis)
        # self.thread.connect(self.worker.do_work())
        



    def init_image_UI(self):
        pass
        # try:
        # print(self.file_list[0][:-3])
        # if self.file_list[0][-3:] == "ply":
            # print("Plyfile found")
            # self.dataset.load_ply(self.get_global_filename(self.current_folder, self.file_list[0]))
            # self.read_ply(self.get_global_filename(self.current_folder, self.file_list[0]))
            # self.image = self.dataset.image
        # else:
        #     self.image = cv2.imread(self.get_global_filename(self.current_folder, self.file_list[0]))
        
        # self.original_img = deepcopy(self.image)
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
        open_file = QAction("Open File", self)

        open_folder = QAction("Open Folder", self)
        open_folder.triggered.connect(lambda: self.get_folder())
        
        file.addAction(open_file)
        file.addAction(open_folder)


        edit = bar.addMenu("Edit")

        edit_toggle = QAction("Toggle Edit Mode", self)
        edit_toggle.triggered.connect(lambda: self.viewer.toggle_edit())

        edit_new_label = QAction("New Label", self)
        edit_new_label.triggered.connect(lambda: self.add_label())

        edit.addAction(edit_toggle)
        edit.addAction(edit_new_label)
        self.setMenuBar(bar)
    
    def add_label(self):
        print("Creating label")
        if self.current_polygon is not None:
            self.current_polygon.creating = False
        # self.dataset.create_polygon(self.get_global_filename(self.current_folder, self.file_list[0]))
        self.dataset.create_polygon(self.selected_file)
        self.create_new_label()
        self.hide_labels()
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
        
        # file = self.file_list_widget.item(0).text()
        self.selected_file = self.file_list_widget.selectedItems()[0].text()
        print(self.selected_file)

        extention = self.file_list[0].split('.')[-1]

        print(extention)
        if extention == "ply" or extention == "projected":
            print("Plyfile found")
            
            self.hide_all()
            self.dataset.load_ply(self.get_global_filename(self.current_folder, self.selected_file))
            self.image = self.dataset.image
            # print(self.dataset.file_labels.keys())
            if self.selected_file in self.dataset.file_labels.keys():
                self.show_selected_file()
            else:
                self.add_label()
            # self.viewer.update_image(self.image)
        else:
            self.image = cv2.imread(self.get_global_filename(self.current_folder, self.selected_file))
        
        self.original_img = deepcopy(self.image)
        
        # We set viewer image to none before setting a new one because none is a case which determines a new original copy
        self.viewer.image = None
        self.viewer.setPhoto(self.image)
        self.viewer.fitInView()
        self.redraw()
        self.label_clicked()
        # self.image = self.viewer.draw_polygons(self.image)
        # self.viewer.update_image(self.image)

        # file = str(QFileDialog.getOpenFileName(self, "Select Directory")[0])
    
    def get_global_filename(self, folder, filename):
        print(str((folder + "/" + filename)))
        return str((folder + "/" + filename))

    # def read_ply(self, filename):
    #     loaded, image, depth, point_pairs, cloud = utils.load_projected(filename)
    #     if loaded:
    #         cloud = utils.map_pairs_2D(image, point_pairs)
    #     else:
    #         plydata = PlyData.read(filename)
    #         image, depth, point_pairs = utils.project_2D(utils.KINECT_AZURE_INTRINSICS, plydata)
    #         cloud = utils.map_pairs_2D(image, point_pairs)
    #         utils.save_projected(image, depth, point_pairs, cloud)
    
    def get_folder(self):
        self.current_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        print(self.current_folder)
        self.file_list = os.listdir(self.current_folder)
        # for file in folder:
        # self.folder_explorer.setText(folder)
        self.file_list_widget.clear()
        self.file_list_widget.addItems(self.file_list)
        self.file_list_widget.repaint()
        self.file_list_widget.show()
        
        
        
        print("Loaded",  self.file_list)


'''
######################################################################
CLASS DATASET


######################################################################
'''
class Dataset():
    def __init__(self):
        # self.labels = []
        # self.polygons = []

        # labels[filename] = (polygons, labels)
        self.file_labels = {}

        self.labels = []
        self.current_polygon = None
        self.parent_polygon = None
        self.sibling_polygons = []
        self.image = None
        self.depth = None
        self.point_pairs = None
        self.cloud = None

    def create_polygon(self, filename):
        label = LabelPolygon()
        label.creating = True
        
        self.current_polygon = label

        self.labels.append(label)
        label.index = len(self.labels)

        #create a file dictionary to keep track of which label exists to which file
        if filename in self.file_labels.keys():
            print("LABELS", self.file_labels.keys())
            # append a new label onto the filelist labels
            temp_labels = self.file_labels[filename]
            temp_labels.append(label.index)
            self.file_labels[filename] = temp_labels
        else:
            print("NEW LABEL")
            # new label encapsulated with list so it can append additional polygons later
            self.file_labels[filename] = [label.index] 
            print("NEW LABEL",label.index, self.file_labels.keys() , self.file_labels[filename])
        
    def export_label(self, polygon):
        pass
        # with open('eggs.csv', 'w', newline='') as csvfile:
        # points = polygon.points

    def display_distance(self):
        points = self.current_polygon.points[-2:]
        dist = utils.get_distance(self.point_pairs, points[0], points[1])
        print(dist)

    def get_polygon(self, index):
        return self.labels[index]
    
    def delete_polygon(self, index):
        self.labels.pop(index)
        self.current_polygon = index-1

    def load_ply(self, filename):
        print("LOADING PLY")
        loaded, self.image, self.depth, self.point_pairs, points, colours = utils.load_projected(filename)
        print(colours)
        print("Loaded Projected:" ,loaded)
        if not loaded:
            print("Did not load projected...")
            plydata = PlyData.read(filename)
            self.image, self.depth, self.point_pairs = utils.project_2D(utils.KINECT_AZURE_INTRINSICS, plydata)
            self.cloud, points, colours = utils.map_pairs_2D(self.image, self.point_pairs)

            utils.save_projected(filename, self.image, self.depth, self.point_pairs, points, colours)
        else:
            print("Did not load projected...")
            self.cloud = utils.numpy_to_o3d(np_cloud_points=points, np_cloud_colors=colours, swap_RGB=True)
            




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

        self.points.append(location)
        self.calculate_mean()

    # def get_points(self, point_idx):
    #     '''
    #     '''

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

    def draw(self, image, infill=(0,0,255,0.5), show_label=True, show_points = True, show_lines = True):
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
                image = cv2.polylines(overlay,[pts], (not self.creating), (0,255,255),1 )
            
            if show_label:
                font_color = (255, 245, 85)
                if self.polygon_label:
                    cv2.putText(output, text=self.polygon_label, org=(int(self.mean[0]) - 20, int(self.mean[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=font_color,thickness=1, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(output, text="No Label", org=(int(self.mean[0]) - 20, int(self.mean[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=font_color,thickness=1, lineType=cv2.LINE_AA)

            
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


'''
######################################################################
CLASS PHOTOVIEWER


######################################################################
'''
class PhotoViewer(QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent, dataset):
        super(PhotoViewer, self).__init__(parent)
        self.parent = parent
        self.dataset = dataset
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
        self.current_polygon = None
        self.setDragMode(QGraphicsView.NoDrag)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # self.update_visualizer()

    def start():
        pass


    @QtCore.pyqtSlot()
    def update_image(self, image):
        '''
        This assignes a numpy image to Qimage and Pixmap.

        Use when done adding elements to the image. (Does not provide clean image, only assignes image passed in to viewer)

        NOTE: Use redraw_image when wanting to work with a clean slate.
        '''
        image = np.nan_to_num(image).astype(np.uint8)
        # self.setGeometry(self.left, self.top, image.shape[1], image.shape[0])
        self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.q_image = QPixmap.fromImage(self.q_image)
        # self.zoom_height = self.q_image.height()
        # self.image_frame.setPixmap(self.q_image)
        self.setPhoto(image, self.q_image)

    def draw_polygons(self, image):
        for polygon in self.dataset.labels:
            if polygon.view is True:
                image = polygon.draw(image)
        return image

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

    @QtCore.pyqtSlot()
    def redraw_image(self):
        '''
        Resets to a copy of the original image (clean slate)

        Used when drawing so objects are not redrawn infinitely. I.E mouse location
        '''
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
        if self.image is None:
            return

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
        if self.image is None:
            return

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

        

        if self.current_polygon is None:
            print(self.parent.selected_file)
            self.parent.add_label()
            # self.dataset.create_polygon(self.parent.selected_file)
            # self.current_polygon = self.dataset.current_polygon

        if self.current_polygon is not None:
            
            if event.button() == Qt.LeftButton and not self.edit_mode:
                print("Left Button Clicked")
                self.current_polygon.assign_point(mapped_point)
                self.redraw_image()
                image = self.current_polygon.draw(self.image)
                
                self.update_image(image)
                self.update_distance(mapped_point)
                
                if self.parent.o3d_vis is not None:
                    print("Visualizer", self.parent.o3d_vis)
                    print("Updating visualizer")
                    
                    points_3d = utils.get_3d_from_pairs(self.current_polygon.points, self.dataset.point_pairs)
                    poly = utils.o3d_polygon(points_3d)
                    utils.o3d_add_object(self.parent.o3d_vis, poly)
                    # self.parent.o3d_vis.update_geometry(poly)
                    # self.parent.o3d_vis.poll_events()
                    # self.parent.o3d_vis.update_renderer()
                    


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

    @QtCore.pyqtSlot()
    def draw_points(self):
        if self.image is None:
            return

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

        if self.image is None:
            return

        # mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))
        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())

        self.drag_start = False
        print("dropped")


        
        if self.current_polygon is not None:
            if self.selected_point is not None and self.edit_mode:
                self.redraw_image()
                self.current_polygon.edit_point(self.selected_point, mapped_point)

                
            # image = self.current_polygon.draw(self.image)
            image = self.draw_polygons(self.image)
            self.update_image(image)
                
            self.selected_point = None

        # Update 3D viewer while dragging and editing (After we update the changed point)
        if self.edit_mode is True and self.parent.o3d_vis is not None:
            cleard = self.parent.o3d_vis.clear_geometries()
            print("Cleared: ", cleard)
            utils.o3d_add_object(self.parent.o3d_vis, [self.parent.dataset.cloud])
            points_3d = utils.get_3d_from_pairs(self.current_polygon.points, self.dataset.point_pairs)
            poly = utils.o3d_polygon(points_3d)
            utils.o3d_add_object(self.parent.o3d_vis, poly)
            




    @QtCore.pyqtSlot()
    def mouseMoveEvent(self, event):

        if self.image is None:
            return

        self.mouse_pos = (event.x(), event.y())
        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())
        self.update_distance(mapped_point)

        try:
            if self.parent.worker.continue_run is False:
                self.parent.o3d.destroy_window()
        except:
            pass

        try:
            image = np.nan_to_num(self.original_img).astype(np.uint8)
            b,g,r = image[mapped_point[1]][mapped_point[0]]

            self.redraw_image()
            image = cv2.circle(image, mapped_point, 1, (int(255-b),int(255-g),int(255-r)))

            if mapped_point in self.dataset.point_pairs.keys():
                point, colour = self.dataset.point_pairs[mapped_point]
                print(point[2])
                image = cv2.putText(image, text=(str((point[2]/1000)) + "m"), org=mapped_point, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=(int(255-b),int(255-g),int(255-r)),thickness=1, lineType=cv2.LINE_AA)
            # image = cv2.rectangle(self.image, mapped_point, mapped_point, (255-b,255-g,255-r))
            # image = self.current_polygon.draw(image)
            image = self.draw_polygons(image)
            # image = cv2.circle(self.image, self.hover_point, 1, (0,255,255), 1)
            self.update_image(image)
        except Exception as e:
            print(e)
            print("Could not grab mapped point.")

        if event.buttons() == QtCore.Qt.NoButton:

            if self.current_polygon is not None:
                #Check when hovering is within distance value to change colour
                point = self.current_polygon.check_near_point(mapped_point, dist=2)

                #When hovering over point
                if point is not None:
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
            print(mapped_point)
            print("dragging")
            self.update_distance(mapped_point, editing=True)
            if self.current_polygon is not None:
                #Check if hover is within distance for assigning a point (distance between new points, used for hold and drag assignment)
                point = self.current_polygon.check_near_point(mapped_point,dist=5)

                if point is None and not self.edit_mode:
                    self.redraw_image()
                    self.current_polygon.assign_point(mapped_point)
                    image = self.current_polygon.draw(self.image)
                    self.update_image(image)



            #If edit mode and point is selected
            if self.edit_mode is True and self.selected_point is not None:
                adjacent_points = self.current_polygon.get_adjacent(self.selected_point)
                #if there is an adjacent point
                if adjacent_points[0] is not None or adjacent_points[1] is not None:
                    self.redraw_image()
                    image = self.current_polygon.draw(self.image)
                    if adjacent_points[0] is not None:
                        image = cv2.line(image, adjacent_points[0], mapped_point, (0,255,255), 1)
                    if adjacent_points[1]is not None:
                        image = cv2.line(image, adjacent_points[1], mapped_point, (0,255,255), 1)
                    self.update_image(image)
        
                

        # else:
            # image = cv2.circle(self.image, self.hover_point, 4, (0,0,0), 2)
            # self.hover_point = None

    def update_distance(self, mapped_point, editing=False):
        print("updating distance")
        if self.current_polygon is not None:
            if len(self.current_polygon.points) >= 1:
                
                #Measure the total distance
                total_distance = 0
                for index, point in enumerate(self.current_polygon.points):
                    if index+1 < len(self.current_polygon.points):
                        #pixel coordinates
                        p1 = self.current_polygon.points[index]
                        p2 = self.current_polygon.points[index+1]
                        # pixels mapped to depth coordinates (if they are valid)
                        if p1 in self.dataset.point_pairs.keys() and p2 in self.dataset.point_pairs.keys():
                            p1, colour = self.dataset.point_pairs[p1]
                            p2, colour = self.dataset.point_pairs[p2]
                            #total distance is the sum of distance between points
                            total_distance += ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)**(1/2)
                            self.parent.get_current_info_widgets(3).setText(("Total Distance: " + str(total_distance/10) + "cm"))

                # Measure the distance between points
                p2 = mapped_point
                p1 = self.current_polygon.points[-1]
        
                if editing and len(self.current_polygon.points) >= 2:
                    p2 = mapped_point
                    p1 = self.current_polygon.points[-1]
                elif len(self.current_polygon.points) >= 2:
                    p1 = self.current_polygon.points[-2]
                    p2 = self.current_polygon.points[-1]

                if p1 in self.dataset.point_pairs.keys() and p2 in self.dataset.point_pairs.keys():
                    p1, colour = self.dataset.point_pairs[p1]
                    p2, colour = self.dataset.point_pairs[p2]
                    distance = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)**(1/2)
                    print("Distance:", distance/10)
                    # self.parent.get_current_info_widgets(0).setText(("Distance: " + str(distance/10) + "cm"))
                    self.parent.get_current_info_widgets(2).setText(("Distance: " + str(distance/10) + "cm"))



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