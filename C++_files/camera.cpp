#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
  VideoCapture cap(0);

  cap.set(CAP_PROP_FRAME_WIDTH, 800);
  cap.set(CAP_PROP_FRAME_HEIGHT, 480);
  
  if (cap.isOpened() == false)
    {
      cout << "Cannot open the video camera" << endl;
      cin.get(); //wait for any key press
      return -1;
    }

  double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); // get the width of frames of the video
  double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); // get the height

  cout << "Resolution of the video : " << dWidth << " x " << dHeight << endl;

  string window_name = "Camera";
  namedWindow(window_name, WINDOW_NORMAL);
  setWindowProperty(window_name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
  moveWindow(window_name, 80, 0);

  while (true)
    {
      Mat frame;
      bool bSuccess = cap.read(frame); //read new frame from Video

      //break if frame cannot be captured
      if (bSuccess == false)
	{
	  cout << "Camera is disconnected" << endl;
	  cin.get(); //wait for keypress
	  break;
	}

      //show the frame
      imshow(window_name, frame);

      //wait for for 10 ms until any key is pressed.  
      //If the 'Esc' key is pressed, break the while loop.
      //If the any other key is pressed, continue the loop 
      //If any key is not pressed withing 10 ms, continue the loop 
      if (waitKey(10) == 27)
        {
          cout << "Esc key is pressed by user. Stoppig the video" << endl;
          break;
        }
    }
  return 0;
}
