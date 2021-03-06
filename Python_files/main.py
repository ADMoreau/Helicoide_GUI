import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
import shutil
import moviepy
from moviepy.editor import VideoFileClip
import sys
import gi
gi.require_version('Gtk', '3.0')                                                                                                          
from gi.repository import Gtk                                                                                                             
gi.require_version('GdkX11', '3.0')                                                                                                       
from gi.repository import GdkX11                                                                                                         
import vlc


video_data = pd.read_csv('files.csv', header=0)
video_clips = {}
#cwd = os.getcwd()

for i in range(video_data.shape[0]):
    with VideoFileClip(video_data['filepath'][i]) as clip:
        frame_size = clip.get_frame(0).shape
    with moviepy.video.io.VideoFileClip.VideoFileClip(video_data['filepath'][i],
                                                      target_resolution = (int(frame_size[0] / 4), int(frame_size[1] / 4))) as clip:
        clip = clip.subclip(video_data['begin'][i], video_data['end'][i])
        if os.path.isdir(video_data['frame'][i]):
            shutil.rmtree(video_data['frame'][i])
        os.mkdir(video_data['frame'][i])
        image_paths = clip.write_images_sequence(video_data['frame'][i] + "/frame%07d.jpg")
        video_clips[video_data['frame'][i]] = {'images' : sorted(image_paths),
                                               'frame_number' : 0,
                                               'frame_size' : [int(frame_size[0] / 4), int(frame_size[1] / 4)],
                                               'transform_mat' : None}
    
# Load the TFLite model and allocate tensors.
homography_interpreter = tf.lite.Interpreter(model_path='/home/pi/raspi_gui/assets/homography_small.tflite')
homography_interpreter.allocate_tensors()

# Get input and output tensors.
homography_input_details = homography_interpreter.get_input_details()
homography_output_details = homography_interpreter.get_output_details()

#15 rectangles with the centers 24 degrees apart on the circumference of a circle
#and the sides 2 degrees offset from center
inner_r = 100
outer_r = 140
offset = 10
# x = r * sin(theta = offset angle from origin clockwise)
# y = r * cos(theta)

def get_x(r, theta, x_offset = 12.5):
    return x_offset + (r * np.sin(theta * np.pi / 180.))

def get_y(r, theta, y_offset = 50.):
    return y_offset - (r * np.cos(theta * np.pi / 180.)) 

center_points = {'front_left' : np.array([0, 0, 1]),
                 'front_right' : np.array([25, 0, 1]),
                 'back_left' : np.array([0, 100, 1]),
                 'back_right' : np.array([25, 100, 1])}

outer_points = {'frame_a' : {'lft' :np.array([get_x(outer_r, 360 - offset), get_y(outer_r, 360 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, offset), get_y(outer_r, offset), 1])},
                'frame_b' : {'lft' : np.array([get_x(outer_r, 24 - offset), get_y(outer_r, 24 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 24 + offset), get_y(outer_r, 24 + offset), 1])},
                'frame_c' : {'lft' : np.array([get_x(outer_r, 48 - offset), get_y(outer_r, 48 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 48 + offset), get_y(outer_r, 48 + offset), 1])},
                'frame_d' : {'lft' : np.array([get_x(outer_r, 72 - offset), get_y(outer_r, 72 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 72 + offset), get_y(outer_r, 72 + offset), 1])},
                'frame_e' : {'lft' : np.array([get_x(outer_r, 96 - offset), get_y(outer_r, 96 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 96 + offset), get_y(outer_r, 96 + offset), 1])},
                'frame_f' : {'lft' : np.array([get_x(outer_r, 120 - offset), get_y(outer_r, 120 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 120 + offset), get_y(outer_r, 120 + offset), 1])},
                'frame_g' : {'lft' : np.array([get_x(outer_r, 144 - offset), get_y(outer_r, 144 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 144 + offset), get_y(outer_r, 144 + offset), 1])},
                'frame_h' : {'lft' : np.array([get_x(outer_r, 168 - offset), get_y(outer_r, 168 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 168 + offset), get_y(outer_r, 168 + offset), 1])},
                'frame_i' : {'lft' : np.array([get_x(outer_r, 192 - offset), get_y(outer_r, 192 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 192 + offset), get_y(outer_r, 192 + offset), 1])},
                'frame_j' : {'lft' : np.array([get_x(outer_r, 216 - offset), get_y(outer_r, 216 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 216 + offset), get_y(outer_r, 216 + offset), 1])},
                'frame_k' : {'lft' : np.array([get_x(outer_r, 240 - offset), get_y(outer_r, 240 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 240 + offset), get_y(outer_r, 240 + offset), 1])},
                'frame_l' : {'lft' : np.array([get_x(outer_r, 264 - offset), get_y(outer_r, 264 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 264 + offset), get_y(outer_r, 264 + offset), 1])},
                'frame_m' : {'lft' : np.array([get_x(outer_r, 288 - offset), get_y(outer_r, 288 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 288 + offset), get_y(outer_r, 288 + offset), 1])},
                'frame_n' : {'lft' : np.array([get_x(outer_r, 312 - offset), get_y(outer_r, 312 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 312 + offset), get_y(outer_r, 312 + offset), 1])},
                'frame_o' : {'lft' : np.array([get_x(outer_r, 336 - offset), get_y(outer_r, 336 - offset), 1]),
                             'rt' : np.array([get_x(outer_r, 336 + offset), get_y(outer_r, 336 + offset), 1])}}

inner_points = {'frame_a' : {'lft' : np.array([get_x(outer_r - inner_r, 180, x_offset=outer_points['frame_a']['lft'][0]),
                                               get_y(outer_r - inner_r, 180, y_offset=outer_points['frame_a']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 180, x_offset=outer_points['frame_a']['rt'][0]),
                                              get_y(outer_r - inner_r, 180, y_offset=outer_points['frame_a']['rt'][1]), 1])},
                'frame_b' : {'lft' : np.array([get_x(outer_r - inner_r, 204, x_offset=outer_points['frame_b']['lft'][0]),
                                               get_y(outer_r - inner_r, 204, y_offset=outer_points['frame_b']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 204, x_offset=outer_points['frame_b']['rt'][0]),
                                              get_y(outer_r - inner_r, 204, y_offset=outer_points['frame_b']['rt'][1]), 1])},
                'frame_c' : {'lft' : np.array([get_x(outer_r - inner_r, 228, x_offset=outer_points['frame_c']['lft'][0]),
                                               get_y(outer_r - inner_r, 228, y_offset=outer_points['frame_c']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 228, x_offset=outer_points['frame_c']['rt'][0]),
                                              get_y(outer_r - inner_r, 228, y_offset=outer_points['frame_c']['rt'][1]), 1])},
                'frame_d' : {'lft' : np.array([get_x(outer_r - inner_r, 252, x_offset=outer_points['frame_d']['lft'][0]),
                                               get_y(outer_r - inner_r, 252, y_offset=outer_points['frame_d']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 252, x_offset=outer_points['frame_d']['rt'][0]),
                                              get_y(outer_r - inner_r, 252, y_offset=outer_points['frame_d']['rt'][1]), 1])},
                'frame_e' : {'lft' : np.array([get_x(outer_r - inner_r, 276, x_offset=outer_points['frame_e']['lft'][0]),
                                               get_y(outer_r - inner_r, 276, y_offset=outer_points['frame_e']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 276, x_offset=outer_points['frame_e']['rt'][0]),
                                              get_y(outer_r - inner_r, 276, y_offset=outer_points['frame_e']['rt'][1]), 1])},
                'frame_f' : {'lft' : np.array([get_x(outer_r - inner_r, 300, x_offset=outer_points['frame_f']['lft'][0]),
                                               get_y(outer_r - inner_r, 300, y_offset=outer_points['frame_f']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 300, x_offset=outer_points['frame_f']['rt'][0]),
                                              get_y(outer_r - inner_r, 300, y_offset=outer_points['frame_f']['rt'][1]), 1])},
                'frame_g' : {'lft' : np.array([get_x(outer_r - inner_r, 324, x_offset=outer_points['frame_g']['lft'][0]),
                                               get_y(outer_r - inner_r, 324, y_offset=outer_points['frame_g']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 324, x_offset=outer_points['frame_g']['rt'][0]),
                                              get_y(outer_r - inner_r, 324, y_offset=outer_points['frame_g']['rt'][1]), 1])},
                'frame_h' : {'lft' : np.array([get_x(outer_r - inner_r, 348, x_offset=outer_points['frame_h']['lft'][0]),
                                               get_y(outer_r - inner_r, 348, y_offset=outer_points['frame_h']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 348, x_offset=outer_points['frame_h']['rt'][0]),
                                              get_y(outer_r - inner_r, 348, y_offset=outer_points['frame_h']['rt'][1]), 1])},
                'frame_i' : {'lft' : np.array([get_x(outer_r - inner_r, 12, x_offset=outer_points['frame_i']['lft'][0]),
                                               get_y(outer_r - inner_r, 12, y_offset=outer_points['frame_i']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 12, x_offset=outer_points['frame_i']['rt'][0]),
                                              get_y(outer_r - inner_r, 12, y_offset=outer_points['frame_i']['rt'][1]), 1])},
                'frame_j' : {'lft' : np.array([get_x(outer_r - inner_r, 36, x_offset=outer_points['frame_j']['lft'][0]),
                                               get_y(outer_r - inner_r, 36, y_offset=outer_points['frame_j']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 36, x_offset=outer_points['frame_j']['rt'][0]),
                                              get_y(outer_r - inner_r, 36, y_offset=outer_points['frame_j']['rt'][1]), 1])},
                'frame_k' : {'lft' : np.array([get_x(outer_r - inner_r, 60, x_offset=outer_points['frame_k']['lft'][0]),
                                               get_y(outer_r - inner_r, 60, y_offset=outer_points['frame_k']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 60, x_offset=outer_points['frame_k']['rt'][0]),
                                              get_y(outer_r - inner_r, 60, y_offset=outer_points['frame_k']['rt'][1]), 1])},
                'frame_l' : {'lft' : np.array([get_x(outer_r - inner_r, 84, x_offset=outer_points['frame_l']['lft'][0]),
                                               get_y(outer_r - inner_r, 84, y_offset=outer_points['frame_l']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 84, x_offset=outer_points['frame_l']['rt'][0]),
                                              get_y(outer_r - inner_r, 84, y_offset=outer_points['frame_l']['rt'][1]), 1])},
                'frame_m' : {'lft' : np.array([get_x(outer_r - inner_r, 108, x_offset=outer_points['frame_m']['lft'][0]),
                                               get_y(outer_r - inner_r, 108, y_offset=outer_points['frame_m']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 108, x_offset=outer_points['frame_m']['rt'][0]),
                                              get_y(outer_r - inner_r, 108, y_offset=outer_points['frame_m']['rt'][1]), 1])},
                'frame_n' : {'lft' : np.array([get_x(outer_r - inner_r, 132, x_offset=outer_points['frame_n']['lft'][0]),
                                               get_y(outer_r - inner_r, 132, y_offset=outer_points['frame_n']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 132, x_offset=outer_points['frame_n']['rt'][0]),
                                              get_y(outer_r - inner_r, 132, y_offset=outer_points['frame_n']['rt'][1]), 1])},
                'frame_o' : {'lft' : np.array([get_x(outer_r - inner_r, 156, x_offset=outer_points['frame_o']['lft'][0]),
                                               get_y(outer_r - inner_r, 156, y_offset=outer_points['frame_o']['lft'][1]), 1]),
                             'rt' : np.array([get_x(outer_r - inner_r, 156, x_offset=outer_points['frame_o']['rt'][0]),
                                              get_y(outer_r - inner_r, 156, y_offset=outer_points['frame_o']['rt'][1]), 1])}}


class ApplicationWindow(Gtk.Window):

    def __init__(self, MRL):
        Gtk.Window.__init__(self, title="")
        self.player_paused=False
        self.is_player_active = False
        self.connect("destroy",Gtk.main_quit)
        self.set_decorated(False)
        self.MRL = MRL
           
    def show(self):
        self.show_all()
        
    def setup_objects_and_events(self):
        self.playback_button = Gtk.Button()
        self.stop_button = Gtk.Button()
        
        self.play_image = Gtk.Image.new_from_icon_name(
                "gtk-media-play",
                Gtk.IconSize.MENU
            )
        self.pause_image = Gtk.Image.new_from_icon_name(
                "gtk-media-pause",
                Gtk.IconSize.MENU
            )
        self.stop_image = Gtk.Image.new_from_icon_name(
                "gtk-media-stop",
                Gtk.IconSize.MENU
            )
        
        self.playback_button.set_image(self.play_image)
        self.stop_button.set_image(self.stop_image)
        
        self.playback_button.connect("clicked", self.toggle_player_playback)
        self.stop_button.connect("clicked", self.stop_player)
        
        self.draw_area = Gtk.DrawingArea()
        self.draw_area.set_size_request(800,460)
        
        self.draw_area.connect("realize",self._realized)
        
        self.hbox = Gtk.Box(spacing=6)
        self.hbox.pack_start(self.playback_button, True, True, 0)
        self.hbox.pack_start(self.stop_button, True, True, 0)
        
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(self.vbox)
        self.vbox.pack_start(self.draw_area, True, True, 0)
        self.vbox.pack_start(self.hbox, False, False, 0)
        
    def stop_player(self, widget, data=None):
        self.player.stop()
        self.is_player_active = False
        self.playback_button.set_image(self.play_image)
        self.destroy()
        
    def toggle_player_playback(self, widget, data=None):

        """
        Handler for Player's Playback Button (Play/Pause).
        """

        if self.is_player_active == False and self.player_paused == False:
            self.player.play()
            self.playback_button.set_image(self.pause_image)
            self.is_player_active = True

        elif self.is_player_active == True and self.player_paused == True:
            self.player.play()
            self.playback_button.set_image(self.pause_image)
            self.player_paused = False

        elif self.is_player_active == True and self.player_paused == False:
            self.player.pause()
            self.playback_button.set_image(self.play_image)
            self.player_paused = True
        else:
            pass
        
    def _realized(self, widget, data=None):
        self.vlcInstance = vlc.Instance("--no-xlib")
        self.player = self.vlcInstance.media_player_new()
        win_id = widget.get_window().get_xid()
        self.player.set_xwindow(win_id)
        self.player.set_mrl(self.MRL)
        self.player.play()
        self.playback_button.set_image(self.pause_image)
        self.is_player_active = True


def get_h(frame):
    frame = cv2.resize(frame, (128, 128))
    homography_input_data = np.array(frame, dtype=np.float32)
    homography_input_data = np.expand_dims(homography_input_data, axis = 0)
    homography_input_data /= 128
    homography_input_data -= 1
    homography_interpreter.set_tensor(homography_input_details[0]['index'], homography_input_data)
    homography_interpreter.invoke()
    h = homography_interpreter.get_tensor(homography_output_details[0]['index'])[0]
    h = np.append(h, 1)
    h[2] = h[2] * 800
    h[5] = h[5] * 480
    h = h.reshape((3, 3))
    return h

def point_project(h, point, return_all=False):
    point = np.matmul(h, point)
    point = np.transpose(point)
    if return_all:
        return point
    else:
        x = point[0] / point[2]
        y = point[1] / point[2]
        return int(x), int(y)                

class Window():
    
    def __init__(self):
        
        #self.cap = cv2.VideoCapture("/home/pi/MVI_2163_cropped_scaled.MOV")    
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 800)
        self.cap.set(4, 480)
        self.window_name_1 = "Camera"
        self.frame_pos = {}
        cv2.namedWindow(self.window_name_1, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name_1, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self.window_name_1, 80, 0)
        cv2.setMouseCallback(self.window_name_1, self.main_button_callback)
        cv2.setMouseCallback(self.window_name_1, self.main_button_callback)
        self.label_img = cv2.imread("/home/pi/raspi_gui/media/sobel_inverse.jpg")
        self.label_img = cv2.resize(self.label_img, (60, 60))
        try:
            while (self.cap.isOpened()) :
                self.ret, self.frame = self.cap.read()
                self.h_frame = self.frame.copy()
                if self.ret == True:
                    self.break_mutex = False
                    self.frame[420:, :, :] = 255
                    cv2.putText(self.frame, "With", (10, 470), cv2.FONT_HERSHEY_PLAIN, 4, (0), 3)
                    self.frame[420:, 150:210, :] = self.label_img
                    cv2.putText(self.frame, 'In Frame, Press =>',(215,465), cv2.FONT_HERSHEY_PLAIN, 3, (0), 3)
                    self.frame[425:475, 700:795, :] = 0
                    cv2.putText(self.frame, "SCAN", (705, 460), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                    cv2.imshow(self.window_name_1, self.frame)
        
                    if (cv2.waitKey(10)==27) or self.break_mutex == True:
                        print("Escaped")
                        break
                else:
                    break
        except:
            self.cap.release()
            cv2.destroyWindow(self.window_name_1)
    
    def main_button_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x in range(740, 800) and y in range(0, 55):
                self.break_mutex = True
                return
                
            if self.frame_pos != {}:
                if x > 740 and y < 60:
                    self.break_mutex = True
                elif x in range(self.c, self.d) and y in range(self.a, self.b):
                    index = video_data.index[video_data['frame'] == self.max_key][0]
                    MRL = video_data['filepath'][index]
                    window = ApplicationWindow(MRL)
                    window.setup_objects_and_events()
                    window.show()
                    Gtk.main()
                    window.player.stop()
                    window.vlcInstance.release()
                    #del window
                    self.break_mutex = True
                return
                
            else:
                if x in range(700, 800) and y in range(425, 480):                
                    h = get_h(self.h_frame)
                    for temp_frame in outer_points.keys():
                        #reverse the points due to perspective change
                        temp_in_lft = point_project(h, inner_points[temp_frame]['rt'])
                        if temp_in_lft[0] < 0 or temp_in_lft[0] > 799 or temp_in_lft[1] < 0 or temp_in_lft[1] > 479:
                            continue
                        temp_in_rt = point_project(h, inner_points[temp_frame]['lft'])
                        if temp_in_rt[0] < 0 or temp_in_rt[0] > 799 or temp_in_rt[1] < 0 or temp_in_rt[1] > 479:
                            continue
                        temp_out_lft = point_project(h, outer_points[temp_frame]['rt'])
                        if temp_out_lft[0] < 0 or temp_out_lft[0] > 799 or temp_out_lft[1] < 0 or temp_out_lft[1] > 479:
                            continue
                        temp_out_rt = point_project(h, inner_points[temp_frame]['lft'])
                        if temp_out_rt[0] < 0 or temp_out_rt[0] > 799 or temp_out_rt[1] < 0 or temp_out_rt[1] > 479:
                            continue

                        self.frame_pos[temp_frame] = {}
                        self.frame_pos[temp_frame]['tl'] = temp_in_lft
                        self.frame_pos[temp_frame]['tr'] = temp_in_rt
                        self.frame_pos[temp_frame]['bl'] = temp_out_lft
                        self.frame_pos[temp_frame]['br'] = temp_out_rt
                    
                    max_width = 0.0
                    self.max_key = None
                    for key in self.frame_pos.keys():
                        temp_width = np.sqrt((self.frame_pos[key]['tl'][0] - self.frame_pos[key]['tr'][0]) ** 2 +
                                             (self.frame_pos[key]['tl'][1] - self.frame_pos[key]['tr'][1]) ** 2)
                        if temp_width > max_width:
                            center = [(self.frame_pos[key]['tl'][0] + self.frame_pos[key]['tr'][0] / 2),
                                      (self.frame_pos[key]['tl'][1] + self.frame_pos[key]['tr'][1] / 2)]
                            max_width = int(temp_width)
                            self.max_key = key
                    while True:
                        try:
                            #put the warped gif frames into place
                            h_frame_copy = self.h_frame.copy()
                            temp_frame = cv2.imread(video_clips[self.max_key]['images'][video_clips[self.max_key]['frame_number']])
                            new_height = int(temp_frame.shape[1] * (max_width / temp_frame.shape[0]))
                            video_clips[self.max_key]['frame_number'] += 1
                            # if we reach the last frame of the gif, restart the gif
                            if video_clips[self.max_key]['frame_number'] == len(video_clips[self.max_key]['images']) - 1:
                                video_clips[self.max_key]['frame_number'] = 0 
                            temp_frame = cv2.resize(temp_frame, (new_height, max_width))
                            center_height = int(temp_frame.shape[0] / 2)
                            center_width = int(temp_frame.shape[1] / 2)
                            pad_a = 0
                            pad_b = 0
                            pad_c = 0
                            pad_d = 0
                            if center[1] - (temp_frame.shape[0] / 2) < 0:
                                pad_a = 0 - center[1] - (temp_frame.shape[0] / 2)
                            if center[1] + (temp_frame.shape[0] / 2) > 479:
                                pad_b = center[1] + (temp_frame.shape[0] / 2) - 479
                            if center[0] - (temp_frame.shape[1] / 2) < 0:
                                pad_c = 0 - center[0] - (temp_frame.shape[1] / 2)
                            if center[0] + (temp_frame.shape[1] / 2) > 799:
                                pad_d = center[0] + (temp_frame.shape[1] / 2) - 799
                            self.a = int(max(0, center[1] - (temp_frame.shape[0] / 2)))
                            self.b = int(min(479, center[1] + (temp_frame.shape[0] / 2)))
                            self.c = int(max(0, center[0] - (temp_frame.shape[1] / 2)))
                            self.d = int(min(799, center[0] + (temp_frame.shape[1] / 2)))
                            
                            self.h_frame[self.a:self.b, self.c:self.d, :] = temp_frame[int(pad_a):temp_frame.shape[0]-int(pad_b),
                                                                                       int(pad_c):temp_frame.shape[1]-int(pad_d), :]
                            #if words for click prompt
                            #cv2.putText(self.h_frame, "Click Here", (int(center[0]) - 30, int(center[1]) + 10),
                            #            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 3)
                            
                            self.h_frame[5:55, 740:795] = [0, 0, 255]
                            cv2.putText(self.h_frame, "X", (748, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)
                       
                            cv2.imshow(self.window_name_1, self.h_frame)
                            if (cv2.waitKey(10)==27) or self.break_mutex == True:
                                break
                        except Exception as e:
                            print(e)
                            self.break_mutex = True
                            break
        
if __name__ == "__main__":
    cv2.namedWindow("screen_saver", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("screen_saver", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("screen_saver", 80, 0)
    black_frame = np.zeros(shape = (480, 800, 3))
    while True:
        cv2.imshow("screen_saver", black_frame)
        Window()
        if (cv2.waitKey(10)==27):
            cv2.destroyAllWindows()
            break
