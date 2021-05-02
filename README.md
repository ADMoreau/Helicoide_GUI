![Screenshot](media/21:16:28.png?raw=true "Demo")

# GUI

This project works to develop a user interface for immersive artwork utilizing AR, ML and ran on an overclocked (overclocking improves FPS by ~12%) raspberry pi 3B. The program works by using a computer vision model which utilizes tflite model compression to produce a homography matrix that projects a set of anchor points onto an artwork in a circular pattern. From these anchor points the point closest to the camera is selected and from this point a short gif using the frames from files stored in the python directory and selected via the timesteps stored in the csv file to demo a video, which if clicked on plays said video. Homography matrix was used as the artworks was very difficult to use a technique like SLAM to localize the camera position to accomplish similar means. Likewise, the model compression would lead the machine vison model to perform poorly if the points were outside of the training distribution. Once the media has been run, the process is begun again from the beginning.

It is unlikely that models that are this small will work with your project unless you take many steps to minimize variance. Here I am only attempting to recognize a single object for a single frame and will ask for human input to ensure the object is actually in the frame before running the models.

The machine vision model is based on the VGG style whithout batchnorm layers to work with tflite. (Tflite doesn't have the batchnorm layer). Further, during development using C++ to run the tflite models led to strange and unpredictable behavior that did not occur while using python with minimal increase in runtime, which wound up being around a 3/4 a second. Some of the C++ files used are kept in the C++ folder for reference.

# Walkthrough for reproduction on similar datasets

The main component of the artwork that was being identified was a box shaped object. I simply went through the dataset and selected a reasonable set of example images and labelled the box corners that I wanted to use. Then during training I would calculate the homography matrix that would project a proportional rectangle onto the labelled points. During early development SLAM based methods for pose estimation and localization were explored, which required a full 3D model of the scene. For the I used the COLMAP program ran on COLAB GPUs. I have left that file in the preprocess folder.
