# NOTE: Things to fix: 

# Get ./data folder loading automatically, or at least ask for folder to work. (DONE)
# Get rid of blue bar #DONE
# Locate where the data saves #DONE
# Zoom (Bind it to plus/minus as alternative) # DONE
# Command z for deleting last interacted point # DONE
# Redo button




from ctypes import alignment
from datetime import datetime
from fileinput import filename

# import open3d
import cv2
import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QAbstractItemView, QApplication, QAction, QCheckBox,
                             QComboBox, QDockWidget, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSpinBox, QStackedLayout, QStyle,
                             QTextEdit, QVBoxLayout, QWidget, QMenuBar, QGraphicsView, QToolButton, QGraphicsScene, QGraphicsPixmapItem, QFrame, QInputDialog, QProgressBar)
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

import pandas as pd

import webbrowser # To open help menu
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

        # self.
        self.image = None
        # self.graphicsView.setMouseTracking(True)
        self.setMouseTracking(True)
        self.title = 'Label Image'
        self.setWindowIcon(QIcon('Logo_Final (3).png'))
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
         
        # self.current_polygon = None

        self.dataset = Dataset()
        self.label_layouts = []

        self.o3d_vis = None
        self.threads = []

        self.init_buttons()
        self.initUI()
        
        self.user , pressed = QInputDialog.getText(self, "User Name", "Name: ",
                                           QLineEdit.Normal, "")
        while self.user == "":
            self.user , pressed = QInputDialog.getText(self, "User Name", "Name: ",
                                           QLineEdit.Normal, "")
        
        try:
            # path = os.path.abspath("./data/")
            # print(os.getcwd())
            path = os.path.expanduser(os.getcwd()) + "/data/"
            print(path)
            self.get_folder(path)
        except Exception as e:
            print(e)
            print("Could not load data folder, find it by yourself.")
        
        
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

        # utils.display_cloud(self.dataset.cloud, shapes)
        self.threads = [
            self.create_thread(utils.display_cloud(self.dataset.cloud, shapes))
        ]
        for thread in self.threads:
            thread.start()

    def new_visualizer(self):
        self.o3d_viewer.clear_geometries()

    def stop_thread(self):
        self.stop_signal.emit()  # emit the finished signal on stop
        # self.thread.stop()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)


        self.viewer = PhotoViewer(self)
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


        #NOTE PROGRESS BAR WORKS BUT LOOKS WEIRD ON MAC
        # self.load_bar = QProgressBar(self.dock, alignment=Qt.AlignCenter)
        # self.load_bar.setMaximum(100)
        # self.load_bar.setGeometry(60,3,100,15)
        # self.load_bar.setAlignment(Qt.AlignCenter)

        # self.right_body.addWidget(self.load_bar)

        # self.layout.addWidget(self.load_bar)

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

            print(index, active_labels)

            if index not in active_labels:
                self.label_list_widget.item(index).setHidden(True)
            else:
                self.label_list_widget.item(index).setHidden(False)

            # self.label_stack_layout.setCurrentIndex(index)
            # self.current_polygon = self.dataset.labels[index]
        

    def label_clicked(self):
        self.hide_labels()



        index = self.label_list_widget.currentRow()
        self.label_stack_layout.setCurrentIndex(index)
        self.label_stack_layout.setEnabled(True)

        # Show selected stack layout from selected label
        self.label_stack_layout.itemAt(self.label_stack_layout.currentIndex()).widget().setHidden(False)

        print("INDEX:", index)
        # self.current_polygon = self.dataset.labels[index]
        self.dataset.current_polygon = self.dataset.labels[index]
        # self.viewer.current_polygon = self.current_polygon
        self.viewer.redraw_all()

    def create_new_label(self):
        #self.label_stack_layout
        self.hide_labels()

        label_info_layout = QVBoxLayout()
        holder_widget = QWidget() # Add this widget dynamically
        holder_widget.setLayout(label_info_layout)

        self.label_stack_layout.addWidget(holder_widget)
        count = self.label_stack_layout.count() -1
        print("COUNT", count)
        self.label_stack_layout.setCurrentIndex(count)
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
        delete.clicked.connect(lambda: self.delete_pressed())
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
        self.label_list_widget.setCurrentRow(count)
        # self.label_list_widget.item
        # self.label_list_widget.itemSelectionChanged()
        # self.label_list_widget.itemChanged.connect(lambda: self.change_visible(count-1, visible_box=visible))
        
        print(label_name.text())
        label_name.textChanged.connect(lambda: self.handle_label_name_change(label_name, count))
        visible.clicked.connect(lambda: self.change_visible(count, item.checkState(), visible))
        preview.clicked.connect(lambda: self.show_labels_3D())

    def get_current_info_widgets(self, item_at):
        # layout = self.label_stack_layout.children()[0].children()[0].children()
        # print(self.label_stack_layout.currentIndex())
        # if self.label_stack_layout.currentIndex() >= 0:
        holder = self.label_stack_layout.itemAt(self.label_stack_layout.currentIndex()).widget() # We get Holder Widget
        layout = holder.layout()
        item = layout.itemAt(item_at).widget()
        return item
    
    def get_info_widget(self, index ,item_at):
        holder = self.label_stack_layout.itemAt(index).widget() # We get Holder Widget
        layout = holder.layout()
        return layout.itemAt(item_at).widget()
        
    def handle_label_name_change(self, name_widget, label_index):
        print("SELECTED")
        new_text = name_widget.text()
        self.dataset.current_polygon.set_label(new_text)
        self.label_list_widget.item(label_index).setText(new_text)
        self.viewer.redraw_all()

    def hide_all(self):
        for index in range(self.label_list_widget.count()):
            self.get_info_widget(index, 4).setCheckState(False)
            
        for label in self.dataset.labels:
            label.view = False
        
    def show_selected_file(self):
        label_indexes = self.dataset.file_labels[self.selected_file]
        for index in label_indexes:
            label = self.dataset.get_polygon(index)
            label.view = True
            print("Label index", index)
            self.get_info_widget(index, 4).setCheckState(2)

        # self.file_list_widget.selectedItems()[0].text()


    def change_visible(self, index, set_value=None, visible_box=None):
        self.dataset.labels[index].view = not self.dataset.labels[index].view
        self.viewer.redraw_all()
        
    def show_labels_3D(self, index=None):
        '''
        '''
        if self.o3d_vis is None:
            self.start_threads()

    def delete_pressed(self):
        # Get active labels in file and current index in list
        current_index = self.label_stack_layout.currentIndex()
        active_labels = self.dataset.file_labels[self.selected_file]
        print("Current Index", current_index)
        print("Active Label Length", len(active_labels))

        # Get the index of current label in the active_labels list.
        active_index = active_labels.index(current_index)

        holder = self.label_stack_layout.itemAt(current_index).widget() # We get Holder Widget
        layout = holder.layout()
        holder.setParent(None)
        # Delete the polygon in dataset

        # Determine the next layout and label to activate
        next_index = -1
        # if list is larger than 1, we change the index
        if len(active_labels) > 1:
            # If index is the first in the list  set it to the next
            if active_index == 0:
                next_index = active_index + 1
            
            # If the label is the last of the active labels, set it to the second last
            elif active_index == len(active_labels) - 1:
                
                next_index = active_index - 1


        print("Next Index", next_index)

        if next_index >= 0:
            # Update the stack layout with the information the the next indexed polygon
            self.label_stack_layout.setCurrentIndex(next_index)
            self.label_list_widget.setCurrentRow(next_index)
        
        else:
            if self.label_stack_layout.count() > 0:
                widget = holder = self.label_stack_layout.itemAt(self.label_stack_layout.currentIndex()).widget()
                widget.setHidden(True)

            
        # Do this reguardless what the next item is
        self.dataset.delete_polygon(current_index, next_index=next_index)
        #Update viewer polygon to dataset's polygon
        # self.viewer.current_polygon = self.dataset.current_polygon
        # self.current_polygon = self.dataset.current_polygon

        # Hide deleted item
        self.label_list_widget.setRowHidden(current_index, True)
        item = self.label_list_widget.takeItem(current_index)
        del item

    def init_image_UI(self):
        pass

    def init_file_UI(self):
        pass

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
        
        export_all = QAction("Export All", self)
        export_all.triggered.connect(lambda: self.dataset.export_all_labels(self.user, self.current_folder))
        
        file.addAction(open_file)
        file.addAction(open_folder)
        file.addAction(export_all)




        edit = bar.addMenu("Edit")

        edit_undo = QAction("Undo", self)
        edit_undo.triggered.connect(lambda: self.undo())
        edit_undo.setShortcuts([QtGui.QKeySequence(Qt.CTRL + Qt.Key_Z)])

        edit_redo = QAction("Redo", self)
        edit_redo.triggered.connect(lambda: self.redo())
        edit_redo.setShortcuts([QtGui.QKeySequence(Qt.CTRL + Qt.Key_Y)])

        edit_toggle = QAction("Toggle Edit Mode", self)
        edit_toggle.triggered.connect(lambda: self.viewer.toggle_edit())
        edit_toggle.setShortcuts([QtGui.QKeySequence(Qt.Key_E)])

        edit_new_label = QAction("New Label", self)
        edit_new_label.triggered.connect(lambda: self.add_label())

        edit.addAction(edit_undo)
        edit.addAction(edit_redo)
        edit.addAction(edit_toggle)
        edit.addAction(edit_new_label)
        self.setMenuBar(bar)


        view = bar.addMenu("View")

        increase_cursor = QAction("Incerase Cursor", self)
        increase_cursor.triggered.connect(lambda: self.viewer.increase_cursor())
        # increase_cursor.setShortcuts([QtGui.QKeySequence(Qt.Key_Plus)])

        decrease_cursor = QAction("Decrease", self)
        decrease_cursor.triggered.connect(lambda: self.viewer.decrease_cursor())
        # decrease_cursor.setShortcuts([QtGui.QKeySequence(Qt.Key_Minus)])
        

        zoom_in = QAction("Zoom In", self)
        zoom_in.triggered.connect(lambda: self.viewer.zoom_in())
        zoom_in.setShortcuts([QtGui.QKeySequence(Qt.Key_Plus)])
        


        zoom_out = QAction("Zoom Out", self)
        zoom_out.triggered.connect(lambda: self.viewer.zoom_out())
        zoom_out.setShortcuts([QtGui.QKeySequence(Qt.Key_Minus)])

        view.addAction(zoom_in)
        view.addAction(zoom_out)
        

        view.addAction(increase_cursor)
        view.addAction(decrease_cursor)

        help = bar.addMenu("Help")

        wiki = QAction("Help Page", self)
        wiki.triggered.connect(lambda: webbrowser.open('https://github.com/hobbitsyfeet/RemoteAnimalScan/wiki/Buddy-Measure-User-Guide'))

        help.addAction(wiki)


    def undo(self):  
        self.dataset.undo()
        self.viewer.redraw_all()
    
    def redo(self):
        self.dataset.redo()
        self.viewer.redraw_all()

    def add_label(self):
        print("Creating label")
        if self.dataset.current_polygon is not None:
            self.dataset.current_polygon.creating = False
        # self.dataset.create_polygon(utils.get_global_filename(self.current_folder, self.file_list[0]))
        print(self.selected_file)
        self.dataset.create_polygon(self.selected_file)
        self.create_new_label()
        self.hide_labels()
        # self.viewer.set_polygon(self.dataset.current_polygon)

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
        print(self.selected_file[0])
        extention = self.selected_file.split('.')[-1]

        if extention == "ply" or extention == "projected":
            
            self.hide_all()
            self.dataset.load_ply(utils.get_global_filename(self.current_folder, self.selected_file))
            self.image = self.dataset.image
            
            # print(self.dataset.file_labels.keys())
            
            if self.selected_file in self.dataset.file_labels.keys():
                self.show_selected_file()
                #set the current index to the first label in the list
                active_labels = self.dataset.file_labels[self.selected_file]
                # add a label if it is empty to prevent a crash
                if len(active_labels) == 0:
                    self.add_label()

                # print(active_labels[0])
                self.label_list_widget.setCurrentRow(active_labels[0])
                # self.label_stack_layout.setCurrentIndex(active_labels[0]+1)

            else:
                self.add_label()
            

            # self.viewer.update_image(self.image)
        else:
            self.image = cv2.imread(utils.get_global_filename(self.current_folder, self.selected_file))
        
        self.original_img = deepcopy(self.image)
        
        # We set viewer image to none before setting a new one because none is a case which determines a new original copy
        self.viewer.image = None
        self.viewer.setPhoto(self.image)
        self.viewer.fitInView()
        self.viewer.redraw_all()
        self.label_clicked()
        self.viewer.update_o3d_viewer()
        # if self.o3d_vis is not None:
            
        
        # self.image = self.viewer.draw_polygons(self.image)
        # self.viewer.update_image(self.image)

        # file = str(QFileDialog.getOpenFileName(self, "Select Directory")[0])
    
    def get_global_filename(self, folder, filename):
        print(str((folder + "/" + filename)))
        return str((folder + "/" + filename))

    def get_folder(self, foldername = None):
        if foldername is None:
            self.current_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        else:
            self.current_folder = foldername
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

        self.states = []
        self.previous_states = []

        self.labels = []
        self.current_polygon = None
        self.parent_polygon = None
        self.sibling_polygons = []
        self.image = None
        self.depth = None
        self.point_pairs = None
        self.inverse_pairs = None
        self.cloud = None

        self.redone = False
        self.undone = False

        self.dataframe = pd.DataFrame(columns=['Participant',
                                                'Filename',
                                                'Points',
                                                'Distance',
                                                'TotalDistance',
                                                ])

    def undo(self):
        # print("UNDO")
        if self.states:
            self.previous_states.append(self.states[-1])
            polygon = self.states.pop()
            
            print(polygon.points)
            self.labels[polygon.index] = polygon
            self.current_polygon = self.labels[polygon.index]
            # self.previous_states.append(self.current_polygon)
            # self.previous_states.append(self.current_polygon)
            if len(self.previous_states) >= 200:
                self.previous_states.pop(0)

    def redo(self):
        # print("REDO")
        if self.previous_states:
            polygon = self.previous_states.pop()
            print(polygon.points)
            # print(len(self.previous_states))
            self.labels[polygon.index] = polygon
            self.current_polygon = self.labels[polygon.index]
            self.states.append(polygon)
            
            


    def save_state(self):
        print("Saving State")
        self.states.append((deepcopy(self.current_polygon)) )

    def create_polygon(self, filename):
        label = LabelPolygon()
        label.creating = True
        
        self.current_polygon = label

        label.index = len(self.labels)
        self.labels.append(label)
        

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
        
    def export_label(self, polygon, point_pairs, participant):
        '''
        'Participant': Name of participant/user
        'Filename': Filename which polygon resides
        'Points':   Points uses to measure
        'Distance': Distance between first and last point (Euclidean) - important if used only 2 points to measure
        'TotalDistance': Distance between all points accumulated. - Important for measuring the distanece along the surface.                 
        '''
        points = polygon.points
        try:
            distance = utils.get_distance_2D(point_pairs, points[0], points[-1])
        except Exception as e:
            print(e)
            distance = 0

        total_distance = utils.get_total_distance(point_pairs, points)/10
        filename = self.get_filename(polygon)
        
        # csv_path = filename.split('.')[0] + '.csv'
        csv_path = participant + '.csv'
        data = [participant, filename, points, distance, total_distance]

        self.dataframe.loc[0] = data
        try:
            if os.path.isfile(csv_path):
                export_csv = self.dataframe.to_csv (csv_path, index = None, header=False, mode='a')
            else:
                export_csv = self.dataframe.to_csv (csv_path, index = None, header=True, mode='w')
        except Exception as e:
            print(e)
            print("Make sure you do not have", csv_path,  "open.")

    def export_all_labels(self, participant, current_folder):
        for polygon in self.labels:
            filename = self.get_filename(polygon)
            load_file = utils.get_global_filename(current_folder, filename)

            loaded, image, depth, point_pairs,  points, colours = utils.load_projected(load_file)
            print(loaded, point_pairs)
            self.export_label(polygon, point_pairs, participant)

    
    def get_filename(self, polygon):
        filename = None
        for key in self.file_labels.keys():
            if self.labels.index(polygon) in self.file_labels[key]:
                filename = key
                break
        return filename
    
    def get_filename_by_index(self, polygon_index):
        filename = None
        for key in self.file_labels.keys():
            if polygon_index in self.file_labels[key]:
                filename = key
                break
        return filename

    def display_distance(self):
        points = self.current_polygon.points[-2:]
        dist = utils.get_distance(self.point_pairs, points[0], points[1])
        # print(dist)

    def get_polygon(self, index):
        return self.labels[index]
    
    def delete_polygon(self, index, next_index):
        # self.save_state()
        print(next_index)
        #removes index from filelist
        for filename in self.file_labels.keys():
            indices = self.file_labels[filename]




            if index in indices:
                new_indices = self.file_labels[filename]
                new_indices.remove(index)
                print("New indices", new_indices)
                self.file_labels[filename] = new_indices
                # print(self.file_labels[filename])

                # For every index greater than the removed index, reduce index by 1 if we remove an index
                # we use new indices or else we reach a segmentation fault
                for i in new_indices:
                    if i > index:
                        # Item no longer coorelates to index, so we find the item's index
                        item_index = self.file_labels[filename].index(i)
                        self.file_labels[filename][item_index] = self.file_labels[filename][item_index] -1
            

        
        if next_index == -1:
            self.current_polygon = None
        else:
            self.current_polygon = self.labels[next_index]
        self.labels.pop(index)

        # if we deleted all items from filename, delete the key
        if self.file_labels[filename] == []:
            self.file_labels.pop(filename)

        
        
        # self.current_polygon = index-1

    def load_ply(self, filename, progressbar=None):
        loaded, self.image, self.depth, self.point_pairs,  points, colours = utils.load_projected(filename, progressbar)
        if not loaded:
            print("Did not load projected...")
            plydata = PlyData.read(filename)
            print("Calculating Projection...")
            self.image, self.depth, self.point_pairs = utils.project_2D(utils.KINECT_AZURE_INTRINSICS, plydata, progress=progressbar)
            self.cloud, points, colours = utils.map_pairs_2D(self.image, self.point_pairs)

            utils.save_projected(filename, self.image, self.depth, self.point_pairs, points, colours)

        else:
            print("Projection loaded...")
            # progressbar.setFormat("Converting Pointcloud...")
            # progressbar.setValue(60)
            self.cloud = utils.numpy_to_o3d(np_cloud_points=points, np_cloud_colors=colours, swap_RGB=True)
            # progressbar.setValue(100)
            # progressbar.setFormat("Done...")
            




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
        Ends the polygon closing it upself.polygo
        """
        self.queue_save = True
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


'''
######################################################################
CLASS PHOTOVIEWER


######################################################################
'''
class PhotoViewer(QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self.parent = parent
        # self.dataset = dataset
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
        # self.current_polygon = None
        self.cursor_size = 1

        self.setDragMode(QGraphicsView.NoDrag)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # self.update_visualizer()

    def dataset(self):
        return self.parent.dataset

    def start():
        pass

    def increase_cursor(self):
        self.cursor_size += 1
    
    def decrease_cursor(self):
        self.cursor_size -= 1

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
        for polygon in self.dataset().labels:
            if polygon.view is True:
                image = polygon.draw(image)
        return image

    def toggle_edit(self):
        self.edit_mode = not self.edit_mode

    # def set_polygon(self, current_polygon):
        # self.current_polygon = current_polygon

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

    def redraw_all(self):
        self.redraw_image()
        self.update_image(self.draw_polygons(self.image))
        self.update_o3d_viewer()

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
        self.mouse_pos = mapped_point
        # self.map_to_widget(self.image_frame, point)
        
        if self.dataset().current_polygon is None:
            # print(self.parent.selected_file)
            self.parent.add_label()
            # self.dataset.create_polygon(self.parent.selected_file)
            # self.current_polygon = self.dataset.current_polygon

        if self.dataset().current_polygon is not None:
            
            if self.dataset().current_polygon.queue_save is True:
                self.parent.dataset.save_state()
                self.dataset().current_polygon.queue_save = False


            if event.button() == Qt.LeftButton and not self.edit_mode:
                print("Left Button Clicked")
                self.dataset().save_state()
                self.dataset().current_polygon.assign_point(mapped_point)
                self.redraw_image()
                image = self.dataset().current_polygon.draw(self.image)
                
                self.update_image(image)
                self.update_distance(mapped_point)
                
                if self.parent.o3d_vis is not None:
                    print("Visualizer", self.parent.o3d_vis)
                    print("Updating visualizer")
                    
                    points_3d = utils.get_3d_from_pairs(self.dataset().current_polygon.points, self.dataset().point_pairs)
                    poly = utils.o3d_polygon(points_3d)
                    utils.o3d_add_object(self.parent.o3d_vis, poly)
                    self.update_o3d_viewer()
                    # self.parent.o3d_vis.update_geometry(poly)
                    # self.parent.o3d_vis.poll_events()
                    # self.parent.o3d_vis.update_renderer()
                    


            if event.button() == Qt.RightButton:
                print("Right button clicked")
                self.redraw_image()
                self.dataset().save_state()
                point = self.dataset().current_polygon.check_near_point(mapped_point, dist=2)
                self.dataset().current_polygon.remove_point(point)
                self.selected_point = None
                
                image = self.dataset().current_polygon.draw(self.image)
                self.update_image(image)

                '''
                #NOTE this comment block cuts segment and displays it in 3D
                self.current_polygon.end_poly()
                self.redraw_image()
                image = self.current_polygon.draw(self.image)
                mask = self.current_polygon.create_mask(self.image.shape[0], self.image.shape[1])

                cropped , coordinates, cloud = self.current_polygon.get_segment_crop(self.original_img, self.parent.dataset.point_pairs)
                '''
                # self.update_image(image)
            
            #Edit mode select point
            if event.buttons() == QtCore.Qt.LeftButton and self.edit_mode:
                
                #Distance for selected point (should match mouse move event and change colour to indicate accurate selection)
                point = self.dataset().current_polygon.check_near_point(mapped_point, dist=2)
                self.selected_point = point
                

                # self.selected_point = poin

                # self.current_polygon = None
                
    # def paintEvent(self, event):
    # #     print("Paint event")
    #     if self.mouse_pos is not None:
    #         print("painting")
    #         self.draw_points()

    # @QtCore.pyqtSlot()
    # def draw_points(self):
    #     if self.image is None:
    #         return

    #     if self.drag_start:
    #         self._painter.begin(self._scene)
            
    #         self._painter.setRenderHint(QPainter.Antialiasing)
    #         pen = QPen(Qt.red, 7)
    #         brush = QBrush(Qt.red)
    #         self._painter.setPen(pen)
    #         self._painter.setBrush(brush)
    #         point = QtCore.QPoint(self.mouse_pos[0], self.mouse_pos[1])
    #         self._painter.drawPoint(point)
    #         # self._painter.drawPixmap(point)
    #         self._painter.end()
    def zoom_in(self, ammount=1):
        factor = 1.25
        self._zoom += ammount
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    def zoom_out(self, ammount=1):
        factor = 0.8
        self._zoom -= ammount
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    def mouseReleaseEvent(self, event):
        
        if self.image is None:
            return

        # mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))
        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())

        self.drag_start = False
        print("dropped")



        if self.dataset().current_polygon is not None:

            if self.dataset().current_polygon.queue_save is True:
                self.dataset().save_state()
                self.dataset().current_polygon.queue_save = False

            if self.selected_point is not None and self.edit_mode:
                self.redraw_image()
                self.dataset().save_state()
                self.dataset().current_polygon.edit_point(self.selected_point, mapped_point)

            # if self.parent.o3d_vis is not None:
            self.update_o3d_viewer()

            # image = self.current_polygon.draw(self.image)
            image = self.draw_polygons(self.image)
            self.update_image(image)
            self.selected_point = None
            

    def update_o3d_viewer(self):
        if self.parent.o3d_vis is not None:
            cleard = self.parent.o3d_vis.clear_geometries()
            # print("Cleared: ", cleard)
            utils.o3d_add_object(self.parent.o3d_vis, [self.parent.dataset.cloud])

            if self.dataset().current_polygon:
                points_3d = utils.get_3d_from_pairs(self.dataset().current_polygon.points, self.dataset().point_pairs)
                poly = utils.o3d_polygon(points_3d)
                utils.o3d_add_object(self.parent.o3d_vis, poly)

            # Cursor location
            pointer_3d = utils.get_3d_from_pairs([self.mouse_pos], self.dataset().point_pairs)
            if len(pointer_3d) >= 1:
                pointer_box = utils.o3d_box(pointer_3d[0])
                utils.o3d_add_object(self.parent.o3d_vis, [pointer_box])








    @QtCore.pyqtSlot()
    def mouseMoveEvent(self, event):

        if self.image is None:
            return

        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())
        self.mouse_pos = mapped_point

        self.update_distance(mapped_point)

        if self.parent.o3d_vis is not None:
            self.update_o3d_viewer()
        try:
            if self.parent.worker.continue_run is False:
                self.parent.o3d.destroy_window()
        except:
            pass

        try:
            image = np.nan_to_num(self.original_img).astype(np.uint8)
            b,g,r = image[mapped_point[1]][mapped_point[0]]

            self.redraw_image()
            image = cv2.circle(image, mapped_point, radius=self.cursor_size, color=(int(255-b),int(255-g),int(255-r)), thickness=1)

            if mapped_point in self.dataset().point_pairs.keys():
                point, colour = self.dataset().point_pairs[mapped_point]
                # print(point[2])
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

            if self.dataset().current_polygon is not None:
                #Check when hovering is within distance value to change colour
                point = self.dataset().current_polygon.check_near_point(mapped_point, dist=2)

                #When hovering over point
                if point is not None:
                    image = cv2.circle(self.image, point, self.cursor_size, (0,255,0), 1)
                    # image = cv2.circle(self.image, point, 5, (0,0,0), 1)
                    self.hover_point = point
                    self.update_image(image)

                #When hover point is left alone
                elif self.hover_point is not None and point is None:
                    # self.redraw_image()
                    
                    image = cv2.circle(self.image, self.hover_point, self.cursor_size, (0,255,255), 1)
                    # image = cv2.circle(self.image, self.hover_point, 4, (0,0,0), 2)
                    self.hover_point = None

                    self.update_image(image)


        if self.drag_start and event.buttons() != Qt.RightButton:

            print("dragging")
            # if event.button() == Qt.LeftButton:
            if self.dataset().current_polygon is not None:

                if self.dataset().current_polygon.queue_save is True:
                    self.dataset().save_state()
                    self.dataset().current_polygon.queue_save = False

                #Check if hover is within distance for assigning a point (distance between new points, used for hold and drag assignment)
                point = self.dataset().current_polygon.check_near_point(mapped_point,dist=5)

                if point is None and not self.edit_mode:
                    self.redraw_image()
                    self.dataset().current_polygon.assign_point(mapped_point)
                    image = self.dataset().current_polygon.draw(self.image)
                    self.update_image(image)

            #If edit mode and point is selected
            if self.edit_mode is True and self.selected_point is not None:
                adjacent_points = self.dataset().current_polygon.get_adjacent(self.selected_point)
                #if there is an adjacent point
                if adjacent_points[0] is not None or adjacent_points[1] is not None:
                    self.redraw_image()
                    image = self.dataset().current_polygon.draw(self.image)
                    if adjacent_points[0] is not None:
                        image = cv2.line(image, adjacent_points[0], mapped_point, (0,255,255), 1)
                    if adjacent_points[1]is not None:
                        image = cv2.line(image, adjacent_points[1], mapped_point, (0,255,255), 1)
                    self.update_image(image)

            # update distance
            self.update_distance(mapped_point, editing=True)

            # # update 3D view if available
            # if self.parent.o3d_vis is not None:
            #     self.update_o3d_viewer()
                

        # else:
            # image = cv2.circle(self.image, self.hover_point, 4, (0,0,0), 2)
            # self.hover_point = None

    def update_distance(self, mapped_point, editing=False):
        # print("updating distance")
        if self.dataset().current_polygon is not None:
            if len(self.dataset().current_polygon.points) >= 1:

                total_distance = utils.get_total_distance(self.dataset().point_pairs, self.dataset().current_polygon.points)
                # print(total_distance)
                self.parent.get_current_info_widgets(3).setText(("Total Distance: " + str(total_distance/10) + "cm"))       

                # Measure the distance between points
                p2 = mapped_point
                p1 = self.dataset().current_polygon.points[-1]
        
                # if editing and len(self.current_polygon.points) >= 2:
                    # p2 = mapped_point
                    # p1 = self.current_polygon.points[-1]
                # elif len(self.current_polygon.points) >= 2:
                #     p1 = self.current_polygon.points[-2]
                #     p2 = self.current_polygon.points[-1]

                if p1 in self.dataset().point_pairs.keys() and p2 in self.dataset().point_pairs.keys():
                    distance = utils.get_distance_2D(self.dataset().point_pairs, p1, p2)
                    self.parent.get_current_info_widgets(2).setText(("Distance: " + str(distance/10) + "cm"))



def cloud_to_image():
    pass

def image_to_cloud():
    pass

def display_2D():
    pass

def display_3d():
    pass


def label_polygon():
    pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())