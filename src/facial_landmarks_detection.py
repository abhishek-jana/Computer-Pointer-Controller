'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork


class Model_FacialLandmarksDetection:
    '''
    Class for the Facial Landmark Detection Model.
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
        self.inp_name = next(iter(self.net.inputs))
        self.out_name = next(iter(self.net.outputs))
        self.inp_shape = self.net.inputs[self.inp_name].shape
        


    def predict(self, frame):
        processed_frame = self.preprocess_input(frame.copy())
        out_img = self.net_plug.infer({self.inp_name : processed_frame})
        out_img = self.preprocess_output(out_img) 
        ht = frame.shape[0]
        wd = frame.shape[1]        
        out_img = out_img* np.array([wd, ht, wd, ht])
        out_img = out_img.astype(np.int32)        
        lxmin = out_img[0]-15
        lymin = out_img[1]-15
        lxmax = out_img[0]+15
        lymax = out_img[1]+15        
        rxmin = out_img[2]-15
        rymin = out_img[3]-15
        rxmax = out_img[2]+15
        rymax = out_img[3]+15
        l = frame[lymin:lymax, lxmin:lxmax]
        r = frame[rymin:rymax, rxmin:rxmax]        
        eye_dim = [[lxmin,lymin, lxmax, lymax], [rxmin, rymin, rxmax, rymax]]        
        return l, r, eye_dim

    def check_model(self):
        pass
        
    def preprocess_input(self, frame):
        
        h = self.inp_shape[2]
        w = self.inp_shape[3]
        reshaped_frame = cv2.resize(frame, (w, h))
        reshaped_frame = reshaped_frame.transpose((2,0,1))
        reshaped_frame = reshaped_frame.reshape(1, 3, h, w)
        return reshaped_frame
        
    def preprocess_output(self,out):
        
        cell = out[self.out_name][0]
        return  (cell[0][0][0], cell[1][0][0], cell[2][0][0], cell[3][0][0])
