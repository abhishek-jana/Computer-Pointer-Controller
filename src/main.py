
import numpy as np
import time
import os
import cv2

from argparse import ArgumentParser
from input_feeder import InputFeeder

from face_detection import Model_FaceDetection
from facial_landmarks_detection import Model_FacialLandmarksDetection
from gaze_estimation import Model_GazeEstimation
from head_pose_estimation import Model_HeadPoseEstimation
from mouse_controller import MouseController

import warnings
warnings.filterwarnings("ignore")

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help=" Path to input video file or enter CAM for webcam")
    parser.add_argument("-fd", "--facedetection", required=True, type=str,
                        help=" Path to .xml file of trained face detection model.")
    parser.add_argument("-fl", "--faciallandmark", required=True, type=str,
                        help=" Path to .xml file of trained facial landmark detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help=" Path to .xml file of trained head pose estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help=" Path to .xml file of trained gaze gstimation model.")
    
    parser.add_argument("-viz", "--visualize", required=False, nargs='+',
                        default=[],
                        help="Enter --viz fd, fl, hp, ge (Seperated by space)"
                             "for visualizing different model outputs of each frame," 
                              )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="path of shared library")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Enter the target device: "
                             "Options are CPU, GPU, FPGA or MYRIAD. "
                             "(CPU is default)")
    parser.add_argument("-p", "--prob_threshold", required=False, type=float,
                        default=0.5,
                        help="Probability threshold for model detection. Default is 0.5")
    
    
    return parser

def init_models(args):
    """
    Funtion to initialize all models
    """
    detect = Model_FaceDetection(args.facedetection, args.device, args.cpu_extension)
    landmark = Model_FacialLandmarksDetection(args.faciallandmark, args.device, args.cpu_extension)
    gaze = Model_GazeEstimation(args.gazeestimation, args.device, args.cpu_extension)
    pose = Model_HeadPoseEstimation(args.headpose, args.device, args.cpu_extension)
    
    return detect,landmark,gaze,pose

def LoadModel(m1,m2,m3,m4):
    """
    Funtion to load model
    """
    m1.load_model()
    m2.load_model()
    m3.load_model()
    m4.load_model()
          
def inference_frame(m1,m2,m3,m4,inF,args):
    """
    Funtion to calculate frame count and infernece time.
    """
    visualize = args.visualize
    mc = MouseController('high','fast')
    total = 0
    fc = 0
    inf_time = 0
    for ret, frame in inF.next_batch():
        if not ret:
            break;
        if frame is not None:
            fc += 1
            if fc%5 == 0:
                cv2.imshow('video', cv2.resize(frame, (500, 500)))        
            key = cv2.waitKey(60)
            start_inf = time.time()
            face_crop, face_dim = m1.predict(frame.copy(), args.prob_threshold)
            if type(face_crop) == int:
                print("No face detected.")
                if key == 27:
                    break
                continue            
            hp_out = m2.predict(face_crop.copy())            
            le, re, eye_dim = m3.predict(face_crop.copy())            
            new_dim, gv = m4.predict(le, re, hp_out)            
            end_inf = time.time()
            inf_time = inf_time + end_inf - start_inf
            total = total + 1            
            visualization(visualize, frame, face_crop, face_dim, eye_dim, hp_out, gv, le, re)            
            if fc%5 == 0:
                mc.move(new_dim[0], new_dim[1])             
            if key == 27:
                break
    return fc,inf_time
    
     
def visualization(visualize, frame, face_crop, face_dim, eye_dim, hp_out, gv, le, re):
    """
    Function for visualisation (optional)
    """
    if (not len(visualize) == 0):
        bounding_box = frame.copy()                
        if 'fd' in visualize:
            if len(visualize) != 1:
                bounding_box = face_crop
            else:
                cv2.rectangle(bounding_box, (face_dim[0], face_dim[1]), (face_dim[2], face_dim[3]), (0, 150, 0), 3)
        if 'fl' in visualize:
            if not 'fd' in visualize:
                bounding_box = face_crop.copy()
            cv2.rectangle(bounding_box, (eye_dim[0][0], eye_dim[0][1]), (eye_dim[0][2], eye_dim[0][3]), (0,255,0), 3)
            cv2.rectangle(bounding_box, (eye_dim[1][0], eye_dim[1][1]), (eye_dim[1][2], eye_dim[1][3]), (0,255,0), 3)                    
        if 'hp' in visualize:
            cv2.putText(bounding_box,"Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0], hp_out[1], hp_out[2]), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        if 'ge' in visualize:
            if not 'fd' in visualize:
                bounding_box = face_crop.copy()
            mid_x, mid_y, w = int(gv[0] * 12), int(gv[1] * 12), 160                    
            le = cv2.line(le.copy(), (mid_x - w,  mid_y - w), (mid_x + w,  mid_y + w), (0, 0, 255), 2)
            cv2.line(le, (mid_x - w,  mid_y + w), (mid_x + w,  mid_y - w), (0, 0, 255), 2)                    
            re = cv2.line(re.copy(), (mid_x - w,  mid_y - w), (mid_x + w,  mid_y + w), (0, 0, 255), 2)
            cv2.line(re, (mid_x - w,  mid_y + w), (mid_x + w,  mid_y - w), (0, 0, 255), 2)                    
            bounding_box[eye_dim[0][1]:eye_dim[0][3], eye_dim[0][0]:eye_dim[0][2]] = le
            bounding_box[eye_dim[1][1]:eye_dim[1][3], eye_dim[1][0]:eye_dim[1][2]] = re            
    if len(visualize) != 0:
        img_hor = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(bounding_box, (500, 500))))
    else:
        img_hor = cv2.resize(frame, (500, 500))            
    cv2.imshow('Visualization', img_hor)
     


def main():    
    args = build_argparser().parse_args()        
    inputFile = args.input
    inputFeeder = None
    if inputFile.lower() == "cam":
        inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFile):
            print("Unable to find input file")
            exit(1)        
        inputFeeder = InputFeeder("video",inputFile)
    
    start_model_loading = time.time()    
    detect,landmark,gaze,pose=init_models(args)
    inputFeeder.load_data()
    LoadModel(detect, landmark, gaze, pose)
    model_loading_time = time.time() - start_model_loading    
    frame_count,inference_time = inference_frame(detect,pose,landmark,gaze,inputFeeder,args)
    fps = frame_count / inference_time

    print("video is complete!")
    print(f'Model took {model_loading_time} s to load')
    print(f'Inference time of the model is: {inference_time} s')
    print(f'Average inference time of the model is : {inference_time/frame_count} s')
    print(f'FPS is {fps/5} frame/second')

    cv2.destroyAllWindows()
    inputFeeder.close()
     
    
if __name__ == '__main__':
    main()
