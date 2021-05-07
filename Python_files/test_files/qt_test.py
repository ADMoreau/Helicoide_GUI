from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject
import numpy as np
from gpiozero import Button
import time
#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer
import pandas as pd
import moviepy
from moviepy.editor import VideoFileClip
import os
from PIL import Image


from points import inner_points, inner_points_upper
from utils import Projector

class ButtonThread(QObject):
    pressed = pyqtSignal()
    def __init__(self, parent=None):
        super(ButtonThread, self).__init__(parent)
        self.button = Button(12)
        self.button.when_released = self.on_press

    def on_press(self):
        self.pressed.emit()
'''
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._wait_flag = False
        self.send = True
        self.wait = 30
        self.setup(0)

    def setup(self, video_path):# get frames from recorded video
        print(video_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        print(self.video_path)
        print(self.video_path == 0)
        if self.video_path == 0:
            self.player = None
            self.cap.set(3, 800)
            self.cap.set(4, 480)
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            #get audio from recorded video
            try:
                self.player = MediaPlayer(self.video_path)
            except:
                self.player = None
    
    def run(self):
        while self._run_flag:
            while self._wait_flag:
                now = time.time()
                ret, img = self.cap.read()
                if ret and self.send:
                    print("sending original image")
                    self.cv_img = img.copy()
                    self.change_pixmap_signal.emit([img, "CV"])
                    timeDiff = time.time() - now
                    if (timeDiff < 1.0/(self.wait)):
                        time.sleep(1.0/(self.wait) - timeDiff)
                else:
                    self.player = None
                    self.setup(0)
                
                if self.video_path != 0:
                    if self.player == None:
                        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                        #get audio from recorded video
                        try:
                            self.player = MediaPlayer(self.video_path)
                        except:
                            self.player = None
                    elif self.player != None:
                        audio_frame, val = self.player.get_frame()
                        if val != 'eof' and audio_frame is not None:
                            #audio
                            img, t = audio_frame
                    timeDiff = time.time() - now
                    if (timeDiff < 1.0/(self.fps)):
                        time.sleep(1.0/(self.fps) - timeDiff)
            break
        # shut down capture system
        print("Shutting down video cap")
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
'''
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        cap.set(3, 800)
        cap.set(4, 480)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                print("sending original image")
                self.change_pixmap_signal.emit([cv_img, "CV"])
        # shut down capture system
        cap.release()
        self.terminate()
        return

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class HomographyThread(QThread):
    change_pixmap_signal = pyqtSignal(list)
    def __init__(self, video_clips, video_data):
        super().__init__()
        
        self.projector = Projector()
        self.video_clips = video_clips
        self.video_data = video_data
        self.frame_pos = {}

        self._run_flag = False

    def get_transform_mats(self, h):
        """
        get the points projections using homography matrix h
        """
        for temp_frame in inner_points.keys():
            #reverse the points due to perspective change
            temp_in_lft = self.projector.point_project(h, inner_points[temp_frame]['rt'])
            temp_in_rt = self.projector.point_project(h, inner_points[temp_frame]['lft'])

            self.frame_pos[temp_frame] = {}
            self.frame_pos[temp_frame]['tl'] = temp_in_lft
            self.frame_pos[temp_frame]['tr'] = temp_in_rt

        #get the rotation and translation vector that define the camera world position
        rotation_matrix, translation_vector = self.projector.get_rot_and_trans(self.frame_pos)
        def euclidean_dist(pointsA, pointsB):
            return (((pointsA[0] - pointsB[0]) ** 2) +
                    ((pointsA[1] - pointsB[1]) ** 2) +
                    ((pointsA[2] - pointsB[2]) ** 2)) ** 0.5

        #get the closest set of points
        rotM = cv2.Rodrigues(rotation_matrix)[0]
        camera_position = np.array(-np.matrix(rotM).T * np.matrix(translation_vector))
        closest_frame = None
        closest_dist = np.inf
        for p in self.frame_pos.keys():
            for corner in ['lft', 'rt']:
                temp_dist = euclidean_dist(inner_points_upper[p][corner],
                                                camera_position)
                if temp_dist < closest_dist:
                    closest_dist = temp_dist
                    closest_frame = p
            for corner in ['lft', 'rt']:
                temp_dist = euclidean_dist(inner_points[p][corner],
                                                camera_position)
                if temp_dist < closest_dist:
                    closest_dist = temp_dist
                    closest_frame = p

        temp_list = list(inner_points.keys())
        closest_frame_index = temp_list.index(closest_frame)
        #get the transformation matrices for the given video clips, only use 3, 5 runs slowly
        for index in range(closest_frame_index-1, closest_frame_index+2):
            if index >= len(temp_list): 
                i = index % len(temp_list)
            else: i = index
            temp_frame = temp_list[i]
            self.video_clips[temp_frame] = self.projector.get_clip_transform_matrices(self.video_clips[temp_frame],
                                                                                        temp_frame,
                                                                                        rotation_matrix,
                                                                                        translation_vector)
            
        return closest_frame_index
    
    def button_demo(self, cv_img):
        """
        run after scan is requested, and demo the gifs 
        """
        if self._run_flag:
            #create an empty matrix that will store the information for the x, y values of each demo gif
            #later, if reference matrix if screen is tapped to determine which video to play
            print("running button demo")
            self.button_frame = np.empty(shape=(480, 800))
            self.button_frame[:] = np.nan
            temp_list = list(inner_points.keys())               
            #homography matrix to define the plane centered on top of center box
            print("Getting h")
            h = self.projector.get_h(cv_img) 
            print("getting closest frame")
            closest_frame_index = self.get_transform_mats(h)
            #display demo gifs on repeat
            print("playing")
            while self._run_flag:
                try:
                    temp_image = cv_img.copy()
                    for index in range(closest_frame_index-1, closest_frame_index+2):
                        #make sure that referenced gif exists, ie don't use the 15 index in a list len = 13
                        if index >= len(temp_list): 
                            i = index % len(temp_list)
                        else: i = index
                        frame_index = temp_list[i]
                        #open the image file, index defined by 'frame number'
                        temp_frame = cv2.imread(self.video_clips[frame_index]['images'][self.video_clips[frame_index]['frame_number']])
                        #increment the frame number variable, mod to loop instead of ending
                        self.video_clips[frame_index]['frame_number'] = (self.video_clips[frame_index]['frame_number'] + 1) % len(self.video_clips[frame_index]['images'])
                        #warp the frame with the requisite transform_mat
                        warped_img = cv2.warpPerspective(temp_frame,
                                                        np.float32(self.video_clips[frame_index]['transform_mat']),
                                                        (temp_image.shape[1], temp_image.shape[0]))
                        mask = np.zeros((480, 800, 3), dtype=np.uint8)
                        #corners of the gif image in the video frame
                        roi_corners = np.int32(self.video_clips[frame_index]['dst_pts'])
                        #put the gif image in the video
                        cv2.fillConvexPoly(mask, roi_corners, (255, 255, 255))
                        #put the gif index in the button frame reference to use to find out which x, y values 
                        #for which gif was clicked to play video if frame is tapped
                        cv2.fillConvexPoly(self.button_frame, roi_corners, i)
                        mask = cv2.bitwise_not(mask)
                        masked_image = cv2.bitwise_and(temp_image, mask)
                        temp_image = cv2.bitwise_or(warped_img, masked_image)
                    print("sending image to screen")
                    #self.update_image(temp_image)
                    self.change_pixmap_signal.emit([temp_image, "H"])
                except Exception as e:
                    print("Exeption : " + str(e))
                    self.close()

    @pyqtSlot(np.ndarray)
    def recieve_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self._run_flag = True
        self.button_demo(cv_img)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class App(QThread):
    homography_signal = pyqtSignal(np.ndarray)
    image_signal = pyqtSignal(np.ndarray)
    switch = pyqtSignal()
    def __init__(self, video_clips, video_data):
        super().__init__()
        # create the video capture thread
        self.cv_thread = VideoThread()
        # connect its signal to the update_image slot
        self.cv_thread.change_pixmap_signal.connect(self.send_image)
        # start the thread
        self.cv_thread.start()
        self.h_thread = HomographyThread(video_clips, video_data)
        self.h_thread.change_pixmap_signal.connect(self.send_image)
        self.homography_signal.connect(self.h_thread.recieve_image)
        self.h_thread.stop()

        # create the button capture thread
        self.button = ButtonThread()
        # connect its signal to the update_image slot
        self.button.pressed.connect(self.button_press)
        self.button_mutex = 0

    @pyqtSlot(list)
    def send_image(self, params):
        print(params[1], self.button_mutex)
        if params[1] == "CV" and self.button_mutex == 0:
            img = params[0]
            self.img = img.copy()
            self.image_signal.emit(img)
        elif params[1] == "H" and self.button_mutex == 1:
            cv2.imwrite("test.png", params[0])
            self.image_signal.emit(params[0])

    @pyqtSlot()
    def button_press(self):
        print("pressed")
        #switch from webcam to recorded video
        if self.button_mutex == 0:
            self.button_mutex = 1
            self.homography_signal.emit(self.img)
            self.cv_thread.stop()
            #self.cv_thread.exit()
        elif self.button_mutex == 1:
            self.button_mutex = 0
            self.cv_thread.wait = 30

class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.video_clips = {}
        self.video_data = pd.read_csv('/home/pi/Helicoide_GUI/Python_files/data/files.csv', header=0)
        
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        reply = QMessageBox.question(self, '', 'Get Video Frames? (Unnecessary if done before)')
        if reply == QMessageBox.Yes:
            get_frames = True
        if reply == QMessageBox.No:
            get_frames = False

        #get, and if --get_clips set to true, create the frames for the demo gifs
        print("creating clips")
        for i in range(self.video_data.shape[0]):
            with VideoFileClip(self.video_data['filepath'][i]) as clip:
                frame_size = clip.get_frame(0).shape
            if get_frames:
                #open clip at fraction of original resolution
                with moviepy.video.io.VideoFileClip.VideoFileClip(self.video_data['filepath'][i],
                                                            target_resolution = (int(frame_size[0] / 4), int(frame_size[1] / 4))) as clip:
                    #get the clip frames between the beginning and ending points saved in the csv file
                    clip = clip.subclip(self.video_data['begin'][i], self.video_data['end'][i])
                    if os.path.isdir("/home/pi/Helicoide_GUI/Python_files/data/" + self.video_data['frame'][i]):
                        shutil.rmtree("/home/pi/Helicoide_GUI/Python_files/data/" + self.video_data['frame'][i])
                    os.mkdir("/home/pi/Helicoide_GUI/Python_files/data/" + self.video_data['frame'][i])
                    #extract the clips to the designated folder and store a list of the file paths as image_paths
                    image_paths = clip.write_images_sequence("/home/pi/Helicoide_GUI/Python_files/data/" + self.video_data['frame'][i]
                                                                    + "/frame%07d.jpg")
            
            image_paths = ["/home/pi/Helicoide_GUI/Python_files/data/" + self.video_data['frame'][i] + "/" + f for f in os.listdir("/home/pi/Helicoide_GUI/Python_files/data/" + self.video_data['frame'][i])
                                    if os.path.isfile(os.path.join("/home/pi/Helicoide_GUI/Python_files/data/" + self.video_data['frame'][i], f))]
                                                #sorted list of image paths
            self.video_clips[self.video_data['frame'][i]] = {'images' : sorted(image_paths), 
                                                #frame number used later for playing the gif
                                                'frame_number' : 0, 
                                                'frame_size' : [int(frame_size[0] / 4),
                                                                int(frame_size[1] / 4)],
                                                #matrix that will be used later to warp the images for the gifs
                                                'transform_mat' : None, 
                                                #points that will define the transform_mat transformation, destination points in the image frame
                                                'dst_pts': None} 

        self.app = App(self.video_clips, self.video_data)
        self.app.image_signal.connect(self.update_image)
        self.app.switch.connect(self.set_label)

        self.disply_width = 775
        self.display_height = 460
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.set_label()

    @pyqtSlot()
    def set_label(self):
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        p = QPixmap.fromImage(p)
        return p

    def keyPressEvent(self, event):
        """Close application from escape key.

        results in QMessageBox dialog from closeEvent, good but how/why?
        """
        if event.key() == Qt.Key_Escape:
            self.close()
    
    def closeEvent(self, event):
        self.cv_thread.stop()
        event.accept()

    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = Window()
    a.showFullScreen()
    sys.exit(app.exec_())