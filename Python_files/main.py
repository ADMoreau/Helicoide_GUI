from gui import Window
import numpy as np
import cv2
import os
import shutil
import pandas as pd
import moviepy
from moviepy.editor import VideoFileClip
import argparse
 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_clips', action='store_true',
                        help='set --get_clips if a new set of clips needs to be created')
    args = parser.parse_args()

    video_data = pd.read_csv('data/files.csv', header=0)

    #get, and if --get_clips set to true, create the frames for the demo gifs
    video_clips = {}
    
    for i in range(video_data.shape[0]):
        with VideoFileClip(video_data['filepath'][i]) as clip:
            frame_size = clip.get_frame(0).shape
        if args.get_clips == True:
            #open clip at quarter resolution
            with moviepy.video.io.VideoFileClip.VideoFileClip(video_data['filepath'][i],
                                                        target_resolution = (int(frame_size[0] / 4), int(frame_size[1] / 4))) as clip:
                #get the clip frames between the beginning and ending points saved in the csv file
                clip = clip.subclip(video_data['begin'][i], video_data['end'][i])
                if os.path.isdir("data/" + video_data['frame'][i]):
                    shutil.rmtree("data/" + video_data['frame'][i])
                os.mkdir("data/" + video_data['frame'][i])
                #extract the clips to the designated folder and store a list of the file paths as image_paths
                image_paths = clip.write_images_sequence("data/" + video_data['frame'][i]
                                                                + "/frame%07d.jpg")
        else:
            #if not extracting clips, get list of image paths for a given gif
            image_paths = ["data/" + video_data['frame'][i] + "/" + f for f in os.listdir("data/" + video_data['frame'][i])
                                if os.path.isfile(os.path.join("data/" + video_data['frame'][i], f))]
                                            #sorted list of image paths
        video_clips[video_data['frame'][i]] = {'images' : sorted(image_paths), 
                                            #frame number used later for playing the gif
                                            'frame_number' : 0, 
                                            'frame_size' : [int(frame_size[0] / 4),
                                                            int(frame_size[1] / 4)],
                                            #matrix that will be used later to warp the images for the gifs
                                            'transform_mat' : None, 
                                            #points that will define the transform_mat transformation, destination points in the image frame
                                            'dst_pts': None} 

    cv2.namedWindow("screen_saver", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("screen_saver", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("screen_saver", 80, 0)
    black_frame = np.zeros(shape = (480, 800, 3))
    while True:
        cv2.imshow("screen_saver", black_frame)
        Window(video_data, video_clips)
        if (cv2.waitKey(10)==27):
            cv2.destroyAllWindows()
            break