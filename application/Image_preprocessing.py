import numpy as np
import skimage.io as io
import skimage.transform as trans
import os
import numpy as np


class image_preprocessing:
    
    
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]

    COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    
    def adjustData(self, img,mask, flag_multi_class, num_class):
        if(flag_multi_class):
            img = img / 255
            mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
            new_mask = np.zeros(mask.shape + (num_class,))
            for i in range(num_class):
                #for one pixel in the image, find the class in mask and convert it into one-hot vector
                new_mask[mask == i,i] = 1
            new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
            mask = new_mask
        elif(np.max(img) > 1):
            img = img / 255
            mask = mask /255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        return (img,mask)
    
    def testGenerator(self, img, target_size = (256,256), flag_multi_class = False,as_gray = True):
          
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

    def labelVisualize(num_class, color_dict,img):
        img = img[:,:,0] if len(img.shape) == 3 else img
        img_out = np.zeros(img.shape + (3,))
        for i in range(num_class):
            img_out[img == i,:] = color_dict[i]
        return img_out / 255
  

    def saveResult(self, npyfile, flag_multi_class = False, num_class = 2):
        for i,item in enumerate(npyfile):
            img = self.labelVisualize(num_class,self.COLOR_DICT,item) if flag_multi_class else item[:,:,0]
            return img