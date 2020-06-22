import numpy as np
import cv2
import pickle
from tools import Singleton


@Singleton
class NeuralModel:
    def __init__(self):
        #------Loading model for prediction by me  ------
        pickle_in = open("model.pkl","rb")
        self.model = pickle.load(pickle_in)
        print('Neural Network model loaded!!')
        
    def guess(self, image):
        img_rows, img_cols = 28, 28

        img = image.reshape(1,img_rows,img_cols,1)
        #PREDICT
        numberClass = int(self.model.predict_classes(img))
        prediction = self.model.predict(img)
        accVal= np.amax(prediction)
        return (prediction , numberClass , accVal)

