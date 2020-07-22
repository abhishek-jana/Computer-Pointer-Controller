'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork
import math


class Model_GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU',extension = None):

        self.net = None
        self.net_plug = None
        self.inp_name = None
        self.out_name = None
        self.inp_shape = None
        self.out_shape = None

        self.model = model_name
        self.device = device
        self.ext = extension
        self.weights = self.model.split('.')[0]+'.bin'

    def load_model(self,plugin=None):
        if not plugin:
            self.plugin = IECore()
        else:
            self.plugin = plugin

        self.net = IENetwork(model = self.model, weights = self.weights)

        self.net_plug = self.plugin.load_network(network = self.net, device_name = self.device, num_requests = 1)
        self.inp_name = [key for key in self.net.inputs.keys()]
        self.out_name = [key for key in self.net.outputs.keys()]
        self.inp_shape = self.net.inputs[self.inp_name[1]].shape

    def predict(self, l_eye, r_eye, angle):
        lip, rip = self.preprocess_input(l_eye.copy(), r_eye.copy())
        out = self.net_plug.infer({'head_pose_angles':angle, 'left_eye_image':lip, 'right_eye_image':rip})
        dim, eye_loc = self.preprocess_output(out, angle)
        return dim, eye_loc

    def check_model(self):
        pass
        
    def preprocess_input(self, l_eye, r_eye):
        h = self.inp_shape[2]
        w = self.inp_shape[3]
        lreshape = cv2.resize(l_eye, (w, h))
        lreshape = lreshape.transpose((2,0,1))
        lreshape = lreshape.reshape(1, 3, h, w)

        rreshape = cv2.resize(r_eye, (w, h))
        rreshape = rreshape.transpose((2,0,1))
        rreshape = rreshape.reshape(1, 3, h, w)

        return  lreshape,rreshape
        
    def preprocess_output(self, out, angle):        
        c = angle[2]
        cos = math.cos(c*math.pi/180.0)
        sin = math.sin(c*math.pi/180.0)
        eye_loc = out[self.out_name[0]].tolist()[0]
        x_new = eye_loc[0] * cos + eye_loc[1] * sin
        y_new = -eye_loc[0] *  sin+ eye_loc[1] * cos
                
        return (x_new, y_new), eye_loc
