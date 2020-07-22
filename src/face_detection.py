'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork


class Model_FaceDetection:
    '''
    Class for the Face Detection Model.
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

        if not self.ext == None and self.device == 'CPU':
                print("CPU extension added")
                
                self.plugin.add_extension(self.ext, 'CPU')
        
        if self.device == 'CPU':
    
            
                supp_layer = self.plugin.query_network(network = self.net, device_name = self.device)

                no_supp_layer = [lay for lay in self.net.layers.keys() if lay not in supp_layer]
                if len(no_supp_layer) != 0:
                    print("Non supported layers found! Please add another CPU extension ")
                    exit(1)
                
                print("Issue resolved after adding extension")
        else:
            print("CPU extension needed")
            exit(1)

        self.net_plug = self.plugin.load_network(network = self.net, device_name = self.device, num_requests = 1)
        self.inp_name = next(iter(self.net.inputs))
        self.out_name = next(iter(self.net.outputs))
        self.inp_shape = self.net.inputs[self.inp_name].shape

    def predict(self, frame, prob_threshold):
        
        processed_frame = self.preprocess_input(frame.copy())
        out = self.net_plug.infer({self.inp_name : processed_frame})
        out = self.preprocess_output(out, prob_threshold)
        
        if (len(out) == 0):
            return 0, 0
        out = out[0] 
        ht = frame.shape[0]
        wd =frame.shape[1]
        out = out* np.array([wd, ht, wd, ht])
        out = out.astype(np.int32)
        
        cropped = frame[out[1]:out[3], out[0]:out[2]]
        return cropped, out

    def check_model(self):
        pass
        
    def preprocess_input(self, frame):
              
        h = self.inp_shape[2]
        w = self.inp_shape[3]        
        reshaped_frame = cv2.resize(frame, (w, h))
        reshaped_frame = reshaped_frame.transpose((2,0,1))
        reshaped_frame = reshaped_frame.reshape(1, 3, h, w)        
        return reshaped_frame
        
    def preprocess_output(self, out, prob_threshold):
        dim = []
        out = out[self.out_name][0][0]
        for cell in out:
            conf = cell[2]
            
            if conf >= prob_threshold:
                xmin = cell[3]
                xmax = cell[5]
                ymin = cell[4]
                ymax = cell[6]            
                dim.append([xmin, ymin, xmax, ymax])        
        return dim
