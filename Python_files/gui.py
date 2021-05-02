import cv2
import numpy as np
from points import inner_points, inner_points_upper, center_points
from vlc_utils import ApplicationWindow
from utils import Projector
import time
from gpiozero import Button

import gi
gi.require_version('Gtk', '3.0')                                                                                                          
from gi.repository import Gtk                                                                                                             
gi.require_version('GdkX11', '3.0')                                                                                                       
from gi.repository import GdkX11    

#use gpio pin 12 on the pi
button = Button(12)

class Window():
    
    def __init__(self, video_data, video_clips):

        self.video_data = video_data
        self.video_clips = video_clips
        
        #if testing use frames from a prerecorded video
        #self.cap = cv2.VideoCapture("/home/pi/scaled.mov")    
        #otherwise use the attached camera
        self.cap = cv2.VideoCapture(0)

        self.Projector = Projector()

        self.cap.set(3, 800)
        self.cap.set(4, 480)
        self.window_name_1 = "Camera"
        self.frame_pos = {}
        cv2.namedWindow(self.window_name_1, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name_1, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self.window_name_1, 80, 0)
        #window callback if screen is tapped
        cv2.setMouseCallback(self.window_name_1, self.main_button_callback)
        #cv2.setMouseCallback(self.window_name_1, self.main_button_callback)
        
        try:
            while (self.cap.isOpened()) :
                self.ret, self.frame = self.cap.read()
                self.h_frame = self.frame.copy()
                #default start, show frames with scan button
                if self.ret == True:
                    self.break_mutex = False
                    self.frame[425:475, 700:795, :] = 0
                    cv2.putText(self.frame, "SCAN", (705, 460), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                    cv2.imshow(self.window_name_1, self.frame)
                    #if scan button is pressed run routine
                    if button.is_pressed == False:
                        self.button_demo()
                        time.sleep(1)
                    #if escape button is pressed
                    if (cv2.waitKey(10)==27) or self.break_mutex == True:
                        print("Escaped")
                        break
                else:
                    break
        except:
            self.cap.release()
            cv2.destroyWindow(self.window_name_1)

    def get_transform_mats(self, h):
        """
        get the points projections using homography matrix h
        """
        for temp_frame in inner_points.keys():
            #reverse the points due to perspective change
            temp_in_lft = self.Projector.point_project(h, inner_points[temp_frame]['rt'])
            temp_in_rt = self.Projector.point_project(h, inner_points[temp_frame]['lft'])

            self.frame_pos[temp_frame] = {}
            self.frame_pos[temp_frame]['tl'] = temp_in_lft
            self.frame_pos[temp_frame]['tr'] = temp_in_rt

        #get the rotation and translation vector that define the camera world position
        rotation_matrix, translation_vector = self.Projector.get_rot_and_trans(self.frame_pos)
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
            self.video_clips[temp_frame] = self.Projector.get_clip_transform_matrices(self.video_clips[temp_frame],
                                                                                        temp_frame,
                                                                                        rotation_matrix,
                                                                                        translation_vector)
            
        return closest_frame_index

    def button_demo(self):
        """
        run after scan is requested, and demo the gifs 
        """
        #create an empty matrix that will store the information for the x, y values of each demo gif
        #later, if reference matrix if screen is tapped to determine which video to play
        self.button_frame = np.empty(shape=(480, 800))
        self.button_frame[:] = np.nan
        temp_list = list(inner_points.keys())               
        #homography matrix to define the plane centered on top of center box
        h = self.Projector.get_h(self.h_frame) 
        closest_frame_index = self.get_transform_mats(h)
        #display demo gifs on repeat
        while True:
            try:
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
                                                    (self.h_frame.shape[1], self.h_frame.shape[0]))
                    mask = np.zeros((480, 800, 3), dtype=np.uint8)
                    #corners of the gif image in the video frame
                    roi_corners = np.int32(self.video_clips[frame_index]['dst_pts'])
                    #put the gif image in the video
                    cv2.fillConvexPoly(mask, roi_corners, (255, 255, 255))
                    #put the gif index in the button frame reference to use to find out which x, y values 
                    #for which gif was clicked to play video if frame is tapped
                    cv2.fillConvexPoly(self.button_frame, roi_corners, i)
                    mask = cv2.bitwise_not(mask)
                    masked_image = cv2.bitwise_and(self.h_frame, mask)
                    self.h_frame = cv2.bitwise_or(warped_img, masked_image)
                #exit if button is pressed again
                if button.is_pressed == False:
                    #cv2.imwrite(time.strftime("%H:%M:%S", time.gmtime()) + ".png", self.h_frame)
                    self.break_mutex = True
                    time.sleep(1)

                self.h_frame[425:475, 717:795] = [0, 0, 255]
                cv2.putText(self.h_frame, "EXIT", (722, 460), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                #show the edited video frame
                cv2.imshow(self.window_name_1, self.h_frame)
                #escape button on keyboard
                if (cv2.waitKey(10)==27) or self.break_mutex == True:
                    break
            except Exception as e:
                print("Exeption : " + str(e) + " " + str(i))
                self.break_mutex = True
                break
    
    def main_button_callback(self, event, x, y, flags, params):
        """
        function to play the video corresponding to the given gif and x, y values
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            #if x in range(740, 800) and y in range(0, 55):
            if self.frame_pos != {}:
                index = self.button_frame[y, x]
                #index = self.video_data.index[self.video_data['frame'] == self.max_key][0]
                MRL = self.video_data['filepath'][index]
                window = ApplicationWindow(MRL)
                window.setup_objects_and_events()
                window.show()
                Gtk.main()
                window.player.stop()
                window.vlcInstance.release()
                #del window
                self.break_mutex = True
                return
