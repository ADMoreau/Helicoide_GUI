import tensorflow as tf
import numpy as np
import cv2
from points import inner_points, inner_points_upper, outer_points, outer_points_lower


class Projector():
    """
    Class that basically stores opencv ephemera and potentially verbose functions that utilize 
    opencv functionality as well as the functions to get the homography matrix created by the 
    model
    """

    def __init__(self, model_path='/home/pi/homography_small.tflite'):

        # Load the TFLite model and allocate tensors.
        self.homography_interpreter = tf.lite.Interpreter(model_path=model_path)
        self.homography_interpreter.allocate_tensors()

        # Get input and output tensors.
        self.homography_input_details = self.homography_interpreter.get_input_details()
        self.homography_output_details = self.homography_interpreter.get_output_details()

        self.intrinsics = np.array([[800, 0, 400],
                           [0, 800, 240],
                           [0,   0,   1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    def get_h(self, frame):
        """
        get the homography frame using the tflite model
        """
        frame = cv2.resize(frame, (240, 400))
        homography_input_data = np.array(frame, dtype=np.float32)
        homography_input_data = np.expand_dims(homography_input_data, axis = 0)
        homography_input_data /= 128
        homography_input_data -= 1
        self.homography_interpreter.set_tensor(self.homography_input_details[0]['index'], homography_input_data)
        self.homography_interpreter.invoke()
        h = self.homography_interpreter.get_tensor(self.homography_output_details[0]['index'])[0]
        h = np.append(h, 1)
        h[2] = h[2] * 800
        h[5] = h[5] * 480
        h = h.reshape((3, 3))
        return h

    def point_project(self, h, point, return_all=False):
        """
        get the location in the frame for a single 3d point using the homography mat
        """
        #transform h
        point = np.matmul(h, point)
        point = np.transpose(point)
        if return_all:
            return point
        else:
            x = point[0] / point[2]
            y = point[1] / point[2]
            return int(x), int(y)

    def get_rot_and_trans(self, frame_pos):
        """
        function used to determine the camera world postion given world positions and 
        x, y positions in video frame
        """
        image_points = []
        world_points = []

        for point_sets in frame_pos.keys():
            image_points.append(frame_pos[point_sets]['tl'])
            image_points.append(frame_pos[point_sets]['tr'])
            world_points.append(inner_points[point_sets]['rt'])
            world_points.append(inner_points[point_sets]['lft'])

        #print(image_points, world_points)
        (succes, rotation_vector, translation_vector) = cv2.solvePnP(np.array(world_points, dtype=np.float32),
                                                                    np.array(image_points, dtype=np.float32),
                                                                    self.intrinsics,
                                                                    self.dist_coeffs)
            
        return rotation_vector, translation_vector

    def project(self, points, rotation_vector, translation_vector):
        """
        project world points into video frame
        """
        (projected_points, jacobian) = cv2.projectPoints(np.array(points, dtype=np.float32),
                                                         rotation_vector,
                                                         translation_vector,
                                                         self.intrinsics, 
                                                         self.dist_coeffs)
        return projected_points

    def get_clip_transform_matrices(self, video_clip_dict,
                                    temp_frame,
                                    rotation_matrix,
                                    translation_vector):
        """
        function to get the affine transformation matrix that can project the frame of a 
        clip onto the proper location in the image framed by the outer and inner points
        """
        #get the source points dictated by the resolution of the image
        src_pts = np.array([[0, 0],
                            [video_clip_dict['frame_size'][1], 0],
                            [video_clip_dict['frame_size'][1], video_clip_dict['frame_size'][0]],
                            [0, video_clip_dict['frame_size'][0]]]
                            , dtype=np.float32)
        #get the destination points dictated by the projection
        top_rt_p = self.project(inner_points[temp_frame]['lft'],
                                                              rotation_matrix, 
                                                              translation_vector)
        top_lft_p = self.project(inner_points[temp_frame]['rt'],
                                        rotation_matrix, 
                                        translation_vector)
    
        btm_rt_p = self.project(inner_points_upper[temp_frame]['lft'],
                                        rotation_matrix, 
                                        translation_vector)
        btm_lft_p = self.project(inner_points_upper[temp_frame]['rt'],
                                        rotation_matrix, 
                                        translation_vector)
        dst_pts = np.array([[top_lft_p[0][0][0], top_lft_p[0][0][1]],
                               [top_rt_p[0][0][0], top_rt_p[0][0][1]],
                               [btm_rt_p[0][0][0], btm_rt_p[0][0][1]],
                               [btm_lft_p[0][0][0], btm_lft_p[0][0][1]]], dtype=np.float32)
        
        transformation_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        #print(src_pts, dst_pts, transformation_matrix)
        video_clip_dict['transform_mat'] = transformation_matrix
        video_clip_dict['dst_pts'] = dst_pts
        return video_clip_dict