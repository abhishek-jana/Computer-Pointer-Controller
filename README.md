# Computer Pointer Controller
Computer Pointer Controller app is built using Intel OpenVINO toolkit. This app is use to control mouse pointer using our eye and head movement. We can use a recorded video stream or a webcam directly for the input.

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

To do this project, we need to first download the OpenVINO toolkit in our local machine. Instructions can be found [here](https://docs.openvinotoolkit.org/latest/index.html).

To activate the environment open command prompt in windows then:
```
> cd 'C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\bin'
```
And run 'setupvars.bat file'
```
> setupvars.bat
```

####Models

We need to download the following models using the "model downloader" for this app:
 -[Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)

 -[Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html) 
 
 -[Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
 
 -[Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

####Directory Structure

 ```bash

Computer Pointer Controller
├───README.md
├───requirements.txt
├───bin
|   └───demo.mp4
├───model
│   └───intel
│       ├───face-detection-adas-binary-0001
│       │   └───FP32-INT1
│       ├───gaze-estimation-adas-0002
│       │   ├───FP16
│       │   ├───FP16-INT8
│       │   └───FP32
│       ├───head-pose-estimation-adas-0001
│       │   ├───FP16
│       │   ├───FP16-INT8
│       │   └───FP32
│       └───landmarks-regression-retail-0009
│           ├───FP16
│           ├───FP16-INT8
│           └───FP32
└───src
    ├───face_detection.py
    ├───facial_landmarks_detection.py
    ├───gaze_estimation.py
    ├───head_pose_estimation.py
    ├───input_feeder.py
    ├───main.py
    ├───model.py
    └───mouse_controller.py
```    
###Models
We need to download the following models using the "model downloader" for this app:
-[Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
-[Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html) 
-[Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
-[Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)



## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

