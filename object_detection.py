import cv2 as cv
import sys
import numpy as np

def imcv2_recolor(im, a=.1):
    # t = [np.random.uniform()]
    # t += [np.random.uniform()]
    # t += [np.random.uniform()]
    # t = np.array(t) * 2. - 1.
    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    # return np.array(im * 255., np.uint8)
    return im

class object_detector:

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.framework = None
        self.load_model()

    def load_model(self):

        if self.model.endswith('weights') and self.cfg.endswith('cfg'):
            self.net = cv.dnn.readNetFromDarknet(self.cfg, self.model)
            self.framework = 'Darknet'

        else:
            sys.exit('Wrong input for model weights and cfg')


        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
       #  self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
       #  self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


    def predict(self,frame):
        
        # Create a 4D blob from a frame.
        if self.framework == 'Darknet':
            blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        else:
            blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # Run a model
        self.net.setInput(blob)
        out = self.net.forward(ln)
        #print(out)
        return out

        
    def counter(self,p0, out_class, midPoint):
        for mid in midPoint:
            if abs(p0[0] - mid[0]) < 7 and abs(p0[1] - mid[1]) < 7:
                i = midPoint.index(mid)
                print(mid, midPoint, p0, out_class[i])
                return out_class[i]
  
