'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork


class Model_HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
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

        self.net_plug = self.plugin.load_network(network=self.net,device_name=self.device,num_requests=1)
        self.inp_name = next(iter(self.net.inputs))
        self.inp_shape = self.net.inputs[self.inp_name].shape
        self.out_name = next(iter(self.net.outputs))
       

    def predict(self, frame):
        
        processed_frame = self.preprocess_input(frame.copy())        
        out = self.net_plug.infer({self.inp_name : processed_frame})        
        out = self.preprocess_output(out)        
        return out

    def check_model(self):
        pass
        
    def preprocess_input(self, frame):
        
        h = self.inp_shape[2]
        w = self.inp_shape[3]
        
        reshaped_frame = cv2.resize(frame, (w, h))
        reshaped_frame = reshaped_frame.transpose((2,0,1))
        reshaped_frame = reshaped_frame.reshape(1, 3, h, w)

        return reshaped_frame
        
    def preprocess_output(self, out):
        return [out['angle_y_fc'][0,0],out['angle_p_fc'][0,0],out['angle_r_fc'][0,0]]
