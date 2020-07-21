# Computer Pointer Controller
Computer Pointer Controller app is built using Intel OpenVINO toolkit. This app is use to control mouse pointer using our eye and head movement. We can use a recorded video stream or a webcam directly for the input.

## Project Set Up and Installation

To do this project, we need to first download the OpenVINO toolkit in our local machine. Instructions can be found [here](https://docs.openvinotoolkit.org/latest/index.html).

To activate the environment open command prompt in windows then:
```
 cd C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\bin
```
And run 'setupvars.bat file'
```
 setupvars.bat
```

#### Models

We need to download the following models using the "model downloader" for this app:

 -[Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)

 -[Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html) 
 
 -[Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
 
 -[Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

To download the models:

```
 cd C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\tools\model_downloader
```
Then, use the following commands for each models:

**1. Face Detection Model**

```
python downloader.py --name "face-detection-adas-binary-0001"
```

**2. Facial Landmarks Detection Model**

```
python downloader.py --name "landmarks-regression-retail-0009"
```

**3. Head Pose Estimation Model**

```
python downloader.py --name "head-pose-estimation-adas-0001"
```

**4. Gaze Estimation Model**

```
python downloader.py --name "gaze-estimation-adas-0002"
```
This will create an "intel" directory inside the "model_downloader" directory with all four models inside the "intel" directory. We can move the directory to any desired location for convenience.

Next, use the following command:

```
cd <pat to Computer Pointer Controller>
pip install -r requirements.txt
```
To install the dependencies.

#### Directory Structure

The directory structure is as follows:

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

## Demo

To run the model, use the following commands:

**Go to src directory inside project repository**

```
cd <path to projet repo>/src
```

**Run the main.py file**

```
python main.py -f <Path to face detection model .xml file> -fl <Path to facial landmarks detection model .xml file> -hp <Path to head pose estimation model .mxl file> -g <Path to gaze estimation model> -i <Path to input video file or type "CAM" for using webcam> 
```


## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

#### Command line arguments for running the app:

```bash

usage: main.py [-h] -f FACEDETECTIONMODEL -fl FACIALLANDMARKMODEL -hp
               HEADPOSEMODEL -g GAZEESTIMATIONMODEL -i INPUT
               [-flags PREVIEWFLAGS [PREVIEWFLAGS ...]] [-l CPU_EXTENSION]
               [-prob PROB_THRESHOLD] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -f FACEDETECTIONMODEL, --facedetectionmodel FACEDETECTIONMODEL
                        Path to .xml file of Face Detection model.
  -fl FACIALLANDMARKMODEL, --faciallandmarkmodel FACIALLANDMARKMODEL
                        Path to .xml file of Facial Landmark Detection model.
  -hp HEADPOSEMODEL, --headposemodel HEADPOSEMODEL
                        Path to .xml file of Head Pose Estimation model.
  -g GAZEESTIMATIONMODEL, --gazeestimationmodel GAZEESTIMATIONMODEL
                        Path to .xml file of Gaze Estimation model.
  -i INPUT, --input INPUT
                        Path to video file or enter cam for webcam
  -flags PREVIEWFLAGS [PREVIEWFLAGS ...], --previewFlags PREVIEWFLAGS [PREVIEWFLAGS ...]
                        Specify the flags from fd, fld, hp, ge like --flags fd
                        hp fld (Seperated by space)for see the visualization
                        of different model outputs of each frame,fd for Face
                        Detection, fld for Facial Landmark Detectionhp for
                        Head Pose Estimation, ge for Gaze Estimation.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        path of extensions if any layers is incompatible with
                        hardware
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for model to identify the face .
  -d DEVICE, --device DEVICE
                        Specify the target device to run on: CPU, GPU, FPGA or
                        MYRIAD is acceptable. Sample will look for a suitable
                        plugin for device (CPU by default)


```
## Benchmarks

 CPU : Intel i7-8750H CPU 2.20 GHz
 GPU : Intel UHD Graphics 630


| Models                          | Precision | Size    |Model Loading Time | Total Inference Time | Avg Inference Time | FPS   | Hardware |
|---------------------------------|-----------|---------|-------------------|----------------------|--------------------|-------|----------|
| face-detection-adas-binary-0001 | FP32      | 1.86 MB |                   |                      |                    |       |          |
| facial-landmarks-35-adas-0002   | FP32      | 786 KB  |   0.689s          |      1.003s          |    0.017s          | 12FPS |   CPU    |
| head-pose-estimation-adas-0001  | FP32      | 7.34 MB |                   |                      |                    |       |          |
| gaze-estimation-adas-0002       | FP32      | 7.24 MB |                   |                      |                    |       |          |
| Models                          | Precision | Size    |Model Loading Time | Total Inference Time | Avg Inference Time | FPS   | Hardware |
|---------------------------------|-----------|---------|-------------------|----------------------|--------------------|-------|----------|
| face-detection-adas-binary-0001 | FP32      | 1.86 MB |                   |                      |                    |       |          |
| facial-landmarks-35-adas-0002   | FP16      | 413 KB  |   1.19s           |      1.224s          |     0.0207s        | 10FPS |   CPU    |
| head-pose-estimation-adas-0001  | FP16      | 3.69 MB |                   |                      |                    |       |          |
| gaze-estimation-adas-0002       | FP16      | 3.65 MB |                   |                      |                    |       |          |
| Models                          | Precision | Size    |Model Loading Time | Total Inference Time | Avg Inference Time | FPS   | Hardware |
|---------------------------------|-----------|---------|-------------------|----------------------|--------------------|-------|----------|
| face-detection-adas-binary-0001 | FP32      | 1.86 MB |                   |                      |                    |       |          |
| facial-landmarks-35-adas-0002   | INT8      | 314 KB  |   0.95s           |      1.173s          |    0.0199s         | 10FPS |   CPU    |
| head-pose-estimation-adas-0001  | INT8      | 2.05 MB |                   |                      |                    |       |          |
| gaze-estimation-adas-0002       | INT8      | 2.09 MB |                   |                      |                    |       |          |
| Models                          | Precision | Size    |Model Loading Time | Total Inference Time | Avg Inference Time | FPS   | Hardware |
|---------------------------------|-----------|---------|-------------------|----------------------|--------------------|-------|----------|
| face-detection-adas-binary-0001 | FP32      | 1.86 MB |                   |                      |                    |       |          |
| facial-landmarks-35-adas-0002   | FP32      | 786 KB  |  33.77s           |      1.213s          |    0.0205s         | 10FPS |   GPU    |
| head-pose-estimation-adas-0001  | FP32      | 7.34 MB |                   |                      |                    |       |          |
| gaze-estimation-adas-0002       | FP32      | 7.24 MB |                   |                      |                    |       |          |
| Models                          | Precision | Size    |Model Loading Time | Total Inference Time | Avg Inference Time | FPS   | Hardware |
|---------------------------------|-----------|---------|-------------------|----------------------|--------------------|-------|----------|
| face-detection-adas-binary-0001 | FP32      | 1.86 MB |                   |                      |                    |       |          |
| facial-landmarks-35-adas-0002   | FP16      | 413 KB  |  36.04s           |      1.603s          |    0.0272s         |  8FPS |   GPU    |
| head-pose-estimation-adas-0001  | FP16      | 3.69 MB |                   |                      |                    |       |          |
| gaze-estimation-adas-0002       | FP16      | 3.65 MB |                   |                      |                    |       |          |
| Models                          | Precision | Size    |Model Loading Time | Total Inference Time | Avg Inference Time | FPS   | Hardware |
|---------------------------------|-----------|---------|-------------------|----------------------|--------------------|-------|----------|
| face-detection-adas-binary-0001 | FP32      | 1.86 MB |                   |                      |                    |       |          |
| facial-landmarks-35-adas-0002   | INT8      | 314 KB  |  38.49s           |      1.616s          |     0.0274s        | 7FPS  |   GPU    |
| head-pose-estimation-adas-0001  | INT8      | 2.05 MB |                   |                      |                    |       |          |
| gaze-estimation-adas-0002       | INT8      | 2.09 MB |                   |                      |                    |       |          |




## Results

From the table above we see that precision affects the accuracy and inference time. Lower precision gives us better inference time but lose accuracy compared to higher precision. Also, lower precision model takes less memory.

We also see that model loading time and inference time is higher in GPU than in CPU.

## Stand Out Suggestions

This code can take live webcam stream as input video.

### Edge Cases

1. If there is more than one one head in the video, the program ends up selecting one head, but may introduce interference.
2. Sometimes the pointer ends up going to the edge of the screen creating error. It's better to keep the pointer close to the video stream to avoid error. 

