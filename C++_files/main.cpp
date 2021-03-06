#include <iostream>
#include <chrono>
#include <cstdlib>
#include <tuple>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <algorithm>
#include <vector>

#include <opencv2/video/tracking.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

using namespace std;
using namespace cv;
using namespace tflite;
using namespace Eigen;

#define CAPACITY 10;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int img_width = 800;
int img_height = 480;
int img_channels = 3;


unique_ptr<Interpreter> bb_interpreter;

Mat bb_input;
Mat bb_output;

Mat getBoundingBoxTensorMat(int tnum, int debug) {

  TfLiteType t_type = bb_interpreter->tensor(tnum)->type;
  TFLITE_MINIMAL_CHECK(t_type == kTfLiteFloat32);

  TfLiteIntArray* dims = bb_interpreter->tensor(tnum)->dims;
  cout << dims->size << endl;
  if (debug) for (int i = 0; i < dims->size; i++) printf("tensor #%d: %d\n",tnum,dims->data[i]);
  TFLITE_MINIMAL_CHECK(dims->data[0] == 1);
  
  int h = dims->data[1];
  int w = dims->data[2];
  int c = dims->data[3];
  
  float* b_data = bb_interpreter->typed_tensor<float>(tnum);
  TFLITE_MINIMAL_CHECK(b_data != nullptr);

  return Mat(h,w,CV_32FC(c));
}


unique_ptr<Interpreter> homography_interpreter;

Mat homography_input;
Mat homography_output;

Mat getHomographyTensorMat(int tnum, int debug) {

  TfLiteType t_type = homography_interpreter->tensor(tnum)->type;
  TFLITE_MINIMAL_CHECK(t_type == kTfLiteFloat32);

  TfLiteIntArray* dims = homography_interpreter->tensor(tnum)->dims;
  if (debug) for (int i = 0; i < dims->size; i++) printf("tensor #%d: %d\n",tnum,dims->data[i]);
  TFLITE_MINIMAL_CHECK(dims->data[0] == 1);
  
  int h = dims->data[1];
  int w = dims->data[2];
  int c = dims->data[3];

  float* p_data = homography_interpreter->typed_tensor<float>(tnum);
  TFLITE_MINIMAL_CHECK(p_data != nullptr);

  return Mat(h,w,CV_32FC(c),p_data);
}

float front_left_x, front_left_y,
    front_right_x, front_right_y,
    back_left_x, back_left_y,
    back_right_x, back_right_y;
MatrixXd homography_mat(3, 3);
Mat img, input_img, camera_position;

VideoWriter video("test_bb.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(800, 480));

//void callBackFunc(int event, int x, int y, int flags, void* userdata) {
void viewFrame() {
  Mat dist_coeffs(4, 1, DataType<float>::type);
  dist_coeffs.at<float>(0) = 0;
  dist_coeffs.at<float>(1) = 0;
  dist_coeffs.at<float>(2) = 0;
  dist_coeffs.at<float>(3) = 0;
  
  Mat camera_matrix = Mat::eye(3, 3, CV_32F);

  Mat rvec(3, 1, DataType<float>::type);
  Mat rvec2(3, 1, DataType<float>::type);
  Mat rmat(3, 3, DataType<float>::type);
  Mat tvec(3, 1, DataType<float>::type);
    
  vector<Point3f> object_points = {Point3f(-12.5, 50, 0),
				   Point3f(12.5, 50, 0),
				   Point3f(-12.5, -50, 0),
				   Point3f(12.5, -50, 0)};
  vector<Point2f> image_points;

  Mat resized_input;
  resize(input_img, input_img, Size(128, 128));
  input_img.convertTo(bb_input, CV_32FC3, 1.0/128.0, -1.0);
  imshow("Camera", input_img);
  // Run inference
  TFLITE_MINIMAL_CHECK(bb_interpreter->Invoke() == kTfLiteOk);
  //cout << bb_output << endl;
  float bb_x = bb_output.at<float>(0, 0) * 800;
  float bb_y = bb_output.at<float>(0, 1) * 480;
  float bb_width = bb_output.at<float>(0, 2) * 800;
  float bb_height = bb_output.at<float>(0, 3) * 480;
  cout << bb_x << " " << bb_y << " " << bb_width << " " << bb_height << endl;
  /*
  Rect roi(bb_x, bb_y, bb_width, bb_height);
  Mat roi_img = input_img(roi);
  resize(input_img, input_img, Size(128, 128));
  input_img.convertTo(homography_input, CV_32FC3, 1.0/128.0, -1.0);
  
  // Run inference
  TFLITE_MINIMAL_CHECK(homography_interpreter->Invoke() == kTfLiteOk);
  	
  homography_mat(0, 0) = homography_output.at<float>(0);
  homography_mat(0, 1) = homography_output.at<float>(1);
  homography_mat(0, 2) = (homography_output.at<float>(2) * 128) + bb_x;
  homography_mat(1, 0) = homography_output.at<float>(3);
  homography_mat(1, 1) = homography_output.at<float>(4);
  homography_mat(1, 2) = (homography_output.at<float>(5) * 128) + bb_y;
  homography_mat(2, 0) = homography_output.at<float>(6);
  homography_mat(2, 1) = homography_output.at<float>(7);
  homography_mat(2, 2) = 1;
  cout << homography_mat << endl;
  
  Vector3d front_left(0, 0, 1);
  Vector3d front_right(25, 0, 1);
  Vector3d back_left(0, 100, 1);
  Vector3d back_right(25, 100, 1);
      
  front_left = homography_mat * front_left;
  front_left_x = front_left[0] / front_left[2];
  front_left_y = front_left[1] / front_left[2];
              
  front_right =  homography_mat * front_right;
  front_right_x = front_right[0] / front_right[2];
  front_right_y = front_right[1] / front_right[2];
      
  back_left =  homography_mat * back_left;
  back_left_x = back_left[0] / back_left[2];
  back_left_y = back_left[1] / back_left[2];
      
  back_right =  homography_mat * back_right;
  back_right_x = back_right[0] / back_right[2];
  back_right_y = back_right[1] / back_right[2];
	
  image_points = {Point2f(front_left_x, front_left_y),
		  Point2f(front_right_x, front_right_y),
		  Point2f(back_left_x, back_left_y),
		  Point2f(back_right_x, back_right_y)};

  vector<vector<Point3f>> o = {object_points};
  vector<vector<Point2f>> i = {image_points};
  //filler mats not used
  vector<Mat> t;
  vector<Mat> r;
  //used to get the camera intrinsics based on the homography
  calibrateCamera(o, i, input_img.size(), camera_matrix, dist_coeffs, r, t, 0);
  
  //calcualte the actual rotation and translation 
  solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

  Rodrigues(rvec, rmat);
  rmat = rmat.t();
  camera_position = -rmat * tvec;	 

  Point fr = Point(front_right_x, front_right_y);
  Point fl = Point(front_left_x, front_left_y);
  Point br = Point(back_right_x, back_right_y);
  Point bl = Point(back_left_x, back_left_y);
    
  circle(input_img, fr, 10, Scalar(255,0,255), 5);//magenta
  circle(input_img, fl, 10, Scalar(255,255,0), 5);//light blue
  circle(input_img, br, 10, Scalar(0,0,255), 5);//red
  circle(input_img, bl, 10, Scalar(255,0,0), 5);//blue

  vector<float> x = {front_left_x, front_right_x, back_left_x, back_right_x};
  sort(x.begin(), x.end());
  float max_x = x.back();
  float min_x = x.front();
  float width = max_x - min_x;
	  
  vector<float> y = {front_left_y, front_right_y, back_left_y, back_right_y};
  sort(y.begin(), y.end());
  float max_y = y.back();
  float min_y = y.front();
  float height = max_y - min_y;
  */
  
  //string pos = "X rot: " + to_string(camera_position.at<float>(0)) + " Y rot: " + to_string(camera_position.at<float>(1));
  //String size = "Height: " + to_string(height) + " Width: " + to_string(width);
  //putText(input_img, pos, Point(10,30) , FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 126), 2);
  //putText(input_img, size, Point(10,60) , FONT_HERSHEY_SIMPLEX, .75, Scalar(0, 255, 126), 2);
  //video.write(input_img);
  imshow("Camera", input_img);
  
}

int main()
{
  /*
    std::string image_path = samples::findFile("/home/pi/bridge.jpg");
    img = imread(image_path, IMREAD_COLOR);
    cv::cvtColor(img, img, COLOR_BGR2RGB);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
	}*/
  
  //const string homography_model_path = "/home/pi/raspi_gui/assets/homography_bb.tflite";
    const string bb_model_path = "/home/pi/raspi_gui/assets/combined.tflite";

    //unique_ptr<tflite::FlatBufferModel> homography_model =
    //  tflite::FlatBufferModel::BuildFromFile(homography_model_path.c_str());
    //TFLITE_MINIMAL_CHECK(homography_model != nullptr);

    unique_ptr<tflite::FlatBufferModel> bb_model =
      tflite::FlatBufferModel::BuildFromFile(bb_model_path.c_str());
    TFLITE_MINIMAL_CHECK(bb_model != nullptr);

    // Build the interpreter
    //tflite::ops::builtin::BuiltinOpResolver homography_resolver;
    //InterpreterBuilder homography_builder(*homography_model, homography_resolver);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver bb_resolver;
    InterpreterBuilder bb_builder(*bb_model, bb_resolver);
      
    //homography_builder(&homography_interpreter);    
    //TFLITE_MINIMAL_CHECK(homography_interpreter != nullptr);

    bb_builder(&bb_interpreter);    
    TFLITE_MINIMAL_CHECK(bb_interpreter != nullptr);
    
    //homography_interpreter->SetNumThreads(2);
    //homography_interpreter->SetAllowFp16PrecisionForFp32(true);
    
    bb_interpreter->SetNumThreads(2);
    bb_interpreter->SetAllowFp16PrecisionForFp32(true);
    
    // Allocate tensor buffers.
    //TFLITE_MINIMAL_CHECK(homography_interpreter->AllocateTensors() == kTfLiteOk);
    
    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(bb_interpreter->AllocateTensors() == kTfLiteOk);
    
    int debug = 1;
    // get input and output tensor as cv::Mat
    //homography_input = getHomographyTensorMat(homography_interpreter->inputs()[0],debug);
    //homography_output = getHomographyTensorMat(homography_interpreter->outputs()[0],debug);
    
    // get input and output tensor as cv::Mat
    bb_input = getBoundingBoxTensorMat(bb_interpreter->inputs()[0],debug);
    bb_output = getBoundingBoxTensorMat(bb_interpreter->outputs()[0],debug);
    
    homography_input = getBoundingBoxTensorMat(bb_interpreter->inputs()[1],debug);
    homography_output = getBoundingBoxTensorMat(bb_interpreter->outputs()[1],debug);
    
    VideoCapture cap("/home/pi/MVI_2163_cropped_scaled.MOV");

    string windowName = "Camera";
    namedWindow(windowName, WINDOW_NORMAL);
    setWindowProperty(windowName, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    moveWindow(windowName, 80, 0);
    
    using clock = chrono::system_clock;
    using sec = chrono::duration<double>;
    // for milliseconds, use using ms = std::chrono::duration<double, std::milli>
    const auto before = clock::now();
    int frames = 0;
    while(1){
      cap >> img;
      input_img = img;
      frames++;
      if (img.empty()) break;
      
      //setMouseCallback(windowName, callBackFunc);
      viewFrame();
      
      imshow(windowName, input_img);
      if (waitKey(10)==27) break;
    
    }
    
    const sec duration = clock::now() - before;
    float fps = frames / duration.count();
    cout << "fps : " << fps << endl; 
    cap.release();
    destroyAllWindows();
    
    return 0;
}
