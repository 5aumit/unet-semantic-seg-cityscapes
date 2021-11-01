import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

class DataPrep():
    def __init__(self, trainpath, valpath):
        self.trainpath = trainpath
        self.valpath = valpath
        self.x_train = []
        self.y_train = []
        
        
    def create_dataset(self,paths):    
        for path in os.listdir(paths)[1:]:
            img_path = os.path.join(paths,path)
            img = cv2.imread(img_path)
            x_img = img[0:256,0:256]
            y_img = img[0:256,256:512]
            self.x_train.append(x_img)
            self.y_train.append(y_img)
        return np.array(self.x_train), np.array(self.y_train)