#from skimage import measure
from imutils.perspective import four_point_transform
from datetime import datetime
from skimage import exposure
from skimage.color import rgb2hsv, hsv2rgb
import statistics
import torch
from torchvision import models, transforms

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.DenseNet201 import DenseNet201

from sklearn.metrics import classification_report

from PIL import Image, ImageEnhance
from tqdm import tqdm
from cv2 import aruco
import pandas as pd
import numpy as np
import statistics
import cv2 as cv
import imutils
import random
import glob
import pickle
import yaml
import math
import fnmatch, os, re, glob
#from pyzbar.pyzbar import decode
from shapely.geometry.polygon import LinearRing

import imgaug.augmenters as ia
import imageio

    

class Geometry(object):
    
    def __init__(self):
        pass

    def dist_between_two_points(self, p1, p2):
        """
        Method for dist calculation between 2 points in 2D
        INPUT    -> p1, p2: (x1,y1), (x2,y2)
        OUTPUT   -> dist 
        """
        return math.sqrt(math.pow(abs(p1[0] - p2[0]), 2) + math.pow(abs(p1[1] - p2[1]),2))

    def dist_between_two_points_one_axis(self, p1, p2):
        """
        Method for dist calculation between 2 points in 1D
        INPUT    -> p1, p2:  x1,x2
        OUTPUT   -> dist
        """
        return abs(p1[1] - p2[1])

    def point_of_intersection_L2L(self, k1 : float, k2: float, b1: float, b2: float): 
        """
        Method of 2 lines interseption coordinate calculation 
        input  -> x0,y0,z0, first line parameters
               -> x1,y1,z1, second line parameters
        output -> , interseption coordinate 
        """
        M = np.array([[k1, 1],[k2, 1]])
        v = np.array([b1, b2]) 
        #return 
        return (np.linalg.solve(M, v))
     
    def lineEquationFromPoints(self, p1, p2):
        """
        Get equation from 2 points
        #https://taskcode.ru/linear/straight
        """
        k = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p2[1] - k * p2[0]
        #print(" y = %.2f*x + %.2f" % (k, b))
        return round(k, 4), round(b, 4)

    def xy2x_tranformation(self, p1, p2, p3, p4, camera_id = 1):
        """
        Method of transformation from XY to X' system of coordinat for diagonal position
        INPUT     -> p1, p2 - coordinates of markers
                  -> p3     - coordinates of object
                  -> p4     - coordinates of camera    
        """ 
        #markers_diagonal:
        k1, b1 = self.lineEquationFromPoints(p1, p2)
        k2, b2 = self.lineEquationFromPoints(p3, p4)
        x_center, y_center = self.point_of_intersection_L2L(k1 = k1, k2 = k2, b1 = b1, b2 = b2).tolist()
        
        dist = self.dist_between_two_points(p1, p2)

        if (camera_id == 1):
            return round(self.dist_between_two_points(p1, (abs(x_center), abs(y_center)))/dist, 4)
        
        elif (camera_id == 2):
            return self.dist_between_two_points(center, p2)/dist
        
        else:
            return 0

    def xy2x_tranformation_one_axis(self, p_left_nearest, p_right_nearest, p_left_far, p_right_far, vert_center_coord, camera_id = 1):
        """
        Method of transformation from XY to X' system of coordinat for perpendicaular position
        INPUT     -> p1, p4 - coordinates of markers 
        OUTPUT    -> 
        """ 
        
        # CALCULATE DISTANCE BY EDJES COORDINATES FROM HORIZONTAL CAMERA
        dist = self.dist_between_two_points_one_axis(p_left, p_righ)
        
        if (camera_id == 1):
            return (dist - round(self.dist_between_two_points(p_left, vert_center_coord)))/dist

        elif (camera_id == 2):
            #self.dist_between_two_points(vert_center_coord, p_up)/scaled_dist
            return round(self.dist_between_two_points_one_axis(p_left, vert_center_coord)/dist, 4)

        else:
            return 0

    def intersections(self, a, b):
        """
        Method for intersection coordinates calculation  
        """
        ea = LinearRing(a)
        eb = LinearRing(b)
        mp = ea.intersection(eb)

        x = [p.x for p in mp]
        y = [p.y for p in mp]
        return x, y

    def ellipse_polyline(self, ellipses, n = 100):
        """
        Method for ellipse interseption points coordinates calculation
        """
        t = linspace(0, 2 * np.pi, n, endpoint = False)
        st = np.sin(t)
        ct = np.cos(t)
        result = []

        for x0, y0, a, b, angle in ellipses:
            angle = np.deg2rad(angle)
            sa = np.sin(angle)
            ca = np.cos(angle)
            p = np.empty((n, 2))
            p[:, 0] = x0 + a * ca * ct - b * sa * st
            p[:, 1] = y0 + a * sa * ct + b * ca * st
            result.append(p)
        return result

    def ellipse_intersection(self, ellipses):
        #ellipses = [(x1, y1, ra1, rb1, angle1), (x1, y1, ra1, rb1, angle1)]
        a, b = self.ellipse_polyline(ellipses)
        return self.intersections(a, b)

class Image_augmentation(object):
    
    def __init__(self):
        pass

    def flip_horizontal(self, image):
        return cv.flip(image, 1)

    def flip_vertical(self, image):
        return cv.flip(image, 0)

    def channel_shuffle(self, image, param_1):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgba = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
        aug = ia.ChannelShuffle(param_1, channels = [0, 1, 2])
        image_aug = aug(image = rgba)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGBA2RGB)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGB2BGR)
        return image_aug

    def add_value(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgba = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
        aug = ia.Add((-40, 40))
        image_aug = aug(image = rgba)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGBA2RGB)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGB2BGR)
        return image_aug

    def multiply_hue(self, image, mul_saturation=(0.5, 1.2)):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgba = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
        aug = ia.MultiplyHueAndSaturation(mul_saturation)
        image_aug = aug(image = image)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGBA2RGB)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGB2BGR)
        return image_aug

    def sigmoid_contrast(self, image, gain = 5, cutoff = 0.5):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgba = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
        aug = ia.SigmoidContrast(gain, cutoff)
        image_aug = aug(image = image)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGBA2RGB)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGB2BGR)
        return image_aug

    def change_color_temperature(self, image, diapazon = (7000, 10000)):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgba = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
        aug = ia.ChangeColorTemperature(diapazon)
        image_aug = aug(image = image)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGBA2RGB)
        image_aug = cv.cvtColor(image_aug, cv.COLOR_RGB2BGR)
        return image_aug

    def image_horizontal_flip(self, image, flag = True): 
        #NOT FLIP! HERE IS ROTATION!!!!!

        if (flag == True):
            return cv.rotate(image, cv.ROTATE_180)

        else:
            return image

    def image_channel_changing(self, image):
        channel = np.random.randint(3)
        image[:, :, channel] = np.random.randint(256)

        return image

    def image_rotation(self, image, angle = 0):
        '''if (angle == 90):
            return cv.rotate(image, cv.cv2.ROTATE_90_CLOCKWISE)

        elif (angle == 180):
            return cv.rotate(image, cv.ROTATE_180)

        elif (angle == 270):
            return cv.rotate(image, cv.cv2.ROTATE_270_CLOCKWISE)

        else:'''
        angle = int(random.uniform(-angle, angle))
        h, w = image.shape[:2]
        M = cv.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        return cv.warpAffine(image, M, (w, h))

    def image_noise(self, image, max_noise_level = 0.5):
        gauss = np.random.normal(0, random.uniform(0, max_noise_level), image.size)
        gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
        return cv.add(image, gauss)


    def image_blur(self, image, k_param = 3):
        return cv.blur(image, (k_param, k_param ))


    def image_gamma_correction(self, image, gamma = 1.0):
        """Gamma correction method for image processing.  
        # input     -> RGB image and  gamma parameter
        # output    -> image after corrections 
                    -> table variable by looping over all pixel values in the range [0, 255]."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        return cv.LUT(image, table)


    def fill(self, image, h, w):
        return cv.resize(image, (h, w), cv.INTER_CUBIC)


    def image_horizontal_shift(self, image, ratio = 0.0):
        if ratio > 1 or ratio < 0:
            return image

        else:
            ratio = random.uniform(-ratio, ratio)
            h, w = image.shape[:2]
            to_shift = w * ratio

            if ratio > 0:
                image = image[:, :int(w - to_shift), :]

            if ratio < 0:
                image = image[:, int(-1 * to_shift):, :]

            return self.fill(image, h = h, w = w), ratio


    def image_vertical_shift(self, image, ratio = 0.0):
        if ratio > 1 or ratio < 0:
            return image
        
        else:
            ratio = random.uniform(-ratio, ratio)
            h, w = image.shape[:2]
            to_shift = h * ratio

            if ratio > 0:
                image = image[:int(h-to_shift), :, :]
            
            if ratio < 0:
                image = image[int(-1*to_shift):, :, :]

            return self.fill(image, h, w), ratio


    def image_zoom(self, image, zoom_value):
        if (zoom_value > 1 or zoom_value < 0):
            return image

        elif (zoom_value >= 0 and zoom_value <=1):
            # print(zoom_value)
            #zoom_value = random.uniform(zoom_value, 1)
            h, w = image.shape[:2]
            h_taken = int(zoom_value * h)
            w_taken = int(zoom_value * w)
            h_start = random.randint(0, h - h_taken)
            w_start = random.randint(0, w - w_taken)
            image = image[h_start:h_start + h_taken, w_start:w_start + w_taken]

            return self.fill(image, h, w)

    def image_resizing(self, image, scale_percent = 60): 
        h_start, w_start, _ = image.shape
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        #bg = np.zeros((2000,3000,4), dtype = "uint8")
        #trans_mask = bg[:, :, 3] == 0
        #bg[trans_mask] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255]
        image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        return  cv.resize(image, (w_start,h_start), interpolation = cv.INTER_AREA)

    def simple_image_resizing(self, image, scale_percent = 55): 
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv.resize(image, dim, interpolation = cv.INTER_AREA)

    def image_file_resizing(self, image):
        """Method which make smaller """
        height, width, _  = image.shape
        k = random.uniform(1.0, 2.0)
        height = int(height / k) 
        width = int(width / k)
        dim = (width, height)
        new_file = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        height = int(height * k)
        width = int(width * k)
        dim = (width, height)
        return cv.resize(image, dim, interpolation = cv.INTER_AREA)

    def image_trasformation(self, img, pts1, pts2, w = 1000, h = 1000):
        rows,cols,ch = img.shape
        M = cv.getPerspectiveTransform(pts1,pts2)
        return cv.warpPerspective(img,M,(w,h))

    def clean_image_by_mask(self, image, lower, upper, radius = 3, flags = cv.INPAINT_TELEA):
        """ Method for cleaning image elements by color mask"""
        origin_image = image.copy()
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(image, lower, upper)
        output = cv.inpaint(origin_image, mask, radius, flags = flags)
        return output

    def flipped_clone(self, image, label):
        """Method for yolov4 data augmentation. It will flip image and change label files
        input  -> image - cv2 format frame
               -> label - path to label file 
        output -> image - cv2 format frame
               -> dataframe pandas table 
        """
        
        return image, df

    def set_saturation(self, image, persent = 0.35):
        """Method for changing saturation of image
        input  -> image in RGB format
               -> persent of saturation from 0..1
        output -> image in RGB format
        """
        imghsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        (h, s, v) = cv.split(imghsv)
        s = (s * persent).astype(np.uint8)
        imghsv_new = cv.merge([h,s,v])
        return cv.cvtColor(imghsv_new, cv.COLOR_HSV2BGR)

    def set_value(self, image, persent = 0.9):
        """Method for changing saturation of image
        input  -> image in RGB format
               -> persent of saturation from 0..1
        output -> image in RGB format
        """
        imghsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        (h, s, v) = cv.split(imghsv)
        v = (v * persent).astype(np.uint8)
        imghsv_new = cv.merge([h,s,v])
        return cv.cvtColor(imghsv_new, cv.COLOR_HSV2BGR)

    def image_on_image(self, image, scale = 0.5, position_correction = "center"):
        """ Method for simple image augmentation by adding scaled copy over normal size image"""
        # normal size image clone
        if (position_correction == "center"):
            image_copy = image.copy() 
            h,w,_ = image.shape
            h_scaled = int(h * scale)
            w_scaled = int(w * scale)

            y_offset = int(h * pow(scale, 2))
            x_offset = int(w * pow(scale, 2))
            
            image = cv.resize(image, (w_scalsource/yolact/DEKOL/classification/imds_nano_3/TOTAL_DEKOLed, h_scaled), interpolation = cv.INTER_AREA)

            image_copy[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
            
            return image_copy
        else:
            return image

    def image_change_contrast(self, image, factor=0.55):
        """Method for changing contrast
        input  -> image in RGB format
               -> numerical coefficient of contrast changing
                  0 < factor < 1 - decreasing contrast
                  factor > 1     - increasing contrast
        output -> image in RGB format 
        """
        image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(image)
        image_corrected = enhancer.enhance(factor)
        image_corrected = np.array(image_corrected)
        return image_corrected
       
class Net(object):
    
    def __init__(self):
        #self.preprocess = transforms.Compose([transforms.Resize(size = 256), transforms.CenterCrop(size = 224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.model = torch.load('/data/yolact/src/model_17.08.22_1.pth')
        #self.model.eval()
        self.image_train_directory = '/data/yolact/DEKOL/classification/imds_nano_8/train/'
        self.image_val_directory = '/data/yolact/DEKOL/classification/imds_nano_8/val/'
        #self.image_test_directory = '/data/yolact/DEKOL/classification/imds_nano_5/val/'
        #self.image_test_directory = '/data/yolact/DEKOL/classification/imds_nano_3/TOTAL_DEKOL/'
        self.image_test_directory = '/data/yolact/DEKOL/classification/imds_nano_8/test/'
        self.model_path = "/data/yolact/tf_models/"
        self.train_datagen = ImageDataGenerator(
            rescale = 1./255, 
            rotation_range = 40,
            width_shift_range = 0.2, 
            height_shift_range = 0.2, 
            shear_range = 0.2,
            zoom_range = 0.2, 
            horizontal_flip = True, 
            fill_mode = 'nearest') 
            
        self.val_datagen = ImageDataGenerator(rescale = 1 / 255, 
            rotation_range = 25,
            width_shift_range = 0.1, 
            height_shift_range = 0.1, 
            shear_range = 0.1,
            zoom_range = 0.1, 
            horizontal_flip = True
            )
        self.callback_1 = tf.keras.callbacks.EarlyStopping(
            monitor = 'loss',
            min_delta = 0.01,
            mode = "min",
            patience = 3)
        self.callback_2 = tf.keras.callbacks.EarlyStopping(
            monitor = 'acc',
            min_delta = 0.02,
            mode = "max",
            patience = 3)
        self.callback_3 = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            min_delta = 0.01,
            patience = 3,
            mode = "min",
            # baseline = 0.2,
            restore_best_weights = True)
        self.callback_4 = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_acc',
            min_delta = 0.02,
            patience = 3,
            mode = "max",
            restore_best_weights = True)
    

    def keras_model_creation(self, 
        path : str, 
        model_name : str,
        target_size : tuple, 
        info_flag = True):
        """
        Method for keras model saving.
        Needed for preparation of models stack before night training.

        input  -> path [str path for the folder with models]
               -> model_name [str model name in *.h5 format]
               -> target_size [tuple size of input layer]
               -> info_flag [boolean flag for model architecture image saving]
        output -> 0 - all correct / 1 - error

        """
        try:
            folders_list = os.listdir(self.image_train_directory)
            folders_list.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            num_classes = len(folders_list)       

            model = tf.keras.models.Sequential()
            #InceptionV3, VGG16, , DenseNet201, EfficientNetB7
            base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, classes = num_classes, input_shape=(target_size[0],target_size[1], 3))
            base_model.trainable = False
            model.add(base_model)
            model.add(tf.keras.layers.GlobalAveragePooling2D())
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(512, activation = 'relu'))
            model.add(tf.keras.layers.Dense(512, activation = 'relu'))
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            model.save(str(path) + str(model_name))
            
            if info_flag == True:
                plot_model(model, to_file = self.model_path + str("stack_models/") + str(model_name)[:-3] + '.png')
            
            print("[INFO] ML model architecture had saved as image")

            return 0

        except:
            return 1

    
    def stack_keras_models_training(self, 
        models_path : str, 
        images_path : str, 
        batch_size : int, 
        target_size : tuple,
        n_epochs : int, 
        verbose : bool,
        info_flag = True,
        summary = True):
        """
        Method for stack model training. 
        Can train one by one all created models in folders and send on email results.
        input   -> models_path [path to *.h5 models]
                -> images_path [path to *.JPG images]
                -> info_flag [boolean flag for the information by email]
                -> batch_size [int value of bach size]
                -> target_size [tuple value of input layer]
                -> n_epochs [int value of epoch quantity] 
                -> verbose [bool flag of training progress printing]
                -> info_flag [bool, default = True, flag of info printing]
                -> summary [bool, default = True, flag of info sharing with up layer]
        output  -> 0 - all correct / 1 - error
        """

        try:
            print(models_path)
            keras_model_names = fnmatch.filter(os.listdir(models_path), '*.h5') 
            print("[INFO] There are: " + str(len(keras_model_names)) + str(" keras models in ") + str(models_path) + str(" folder."))

            folders_list = os.listdir(self.image_train_directory)
            folders_list.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            num_classes = len(folders_list)
            
            print("[INFO] Setup train and valid generators")
            train_generator = self.train_datagen.flow_from_directory(
                        self.image_train_directory,  # This is the source directory for training images
                        target_size = target_size,  # All images will be resized to 200 x 200
                        batch_size = batch_size,
                        shuffle = True,
                        classes = folders_list,
                        class_mode = 'categorical')

            valid_generator = self.val_datagen.flow_from_directory(
                    self.image_val_directory,  # This is the source directory for training images
                    target_size = target_size,  # All images will be resized to 200 x 200
                    batch_size = batch_size,
                    classes = folders_list,
                    class_mode = 'categorical')
            print("[INFO] n_epoch is: ", n_epochs)
            for i in range(len(keras_model_names)):
                print("[INFO] Starting training of " + str(keras_model_names[i]) + str(" model.") )
                try:
                    model = tf.keras.models.load_model(models_path + keras_model_names[i])
                    
                    model.compile(
                        loss = 'categorical_crossentropy',
                        optimizer = Adam(lr = 0.01),
                        metrics = ['acc'])
                        
                    total_sample = train_generator.n
                    valid_sample = valid_generator.n 
                    
                    if (summary): 
                        #info = model.summary()
                        stringlist = []
                        model.summary(print_fn=lambda x: stringlist.append(x))
                        info = "\n".join(stringlist)
                        print(info)

                    history = model.fit(
                        train_generator, 
                        steps_per_epoch = int(total_sample / batch_size),  
                        epochs = n_epochs,
                        verbose = verbose,
                        validation_data = valid_generator,
                        validation_steps = int(valid_sample / batch_size))#,
                        #callbacks = [self.callback_3])

                    model.save(models_path + str("FIT_") + keras_model_names[i])
                    print("[INFO] Model " + str(keras_model_names[i]) + str(" had saved in folder ") + str(models_path))


                except KeyboardInterrupt:
                    sys.exit(0)

                except Exception as e: print(e)
                    

            print("[INFO] Training has done!")
            return 0

        except:
            return 1
        

    def net_processing(self, image, settings):
        return image
    

    def train_CNN(self, settings, model_name = '', 
        target_size = (200, 200), batch_size = 64, 
        n_epochs = 11, summary = True, verbose = True):
        """
        Method for CNN model training for dekol classification
        input     -> target_size [size of input matrix]
                  -> batch_size [batch_size]
                  -> n_epochs [quantity of iteration]
                  -> summary [writing information about model]
                  -> verbose [writing progress bar of process]
                  -> model_name [name of model file]
        output    -> --
        info      -> https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_flow_from_directory.ipynb
                  -> https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
                  -> https://github.com/anilsathyan7/pytorch-image-classification #torch
                  -> https://www.geeksforgeeks.org/python-image-classification-using-keras/
        """
      
        if model_name == '' : 
            return 1

        else:
            #sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(log_device_placement = True))

            folders_list = os.listdir(self.image_train_directory)
            folders_list.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            num_classes = len(folders_list)

            #data = np.array(data, dtype="float") / 255.0
            #lb = LabelBinarizer()
            #labels = lb.fit_transform(labels)

            # All images will be rescaled by 1./255


            train_generator = self.train_datagen.flow_from_directory(
                    self.image_train_directory,  # This is the source directory for training images
                    target_size = target_size,  # All images will be resized to 200 x 200
                    batch_size = batch_size,
                    shuffle = True,
                    classes = folders_list,
                    class_mode = 'categorical')

            valid_generator = self.val_datagen.flow_from_directory(
                self.image_val_directory,  # This is the source directory for training images
                target_size = target_size,  # All images will be resized to 200 x 200
                batch_size = batch_size,
                classes = folders_list,
                class_mode = 'categorical')

            # tf.keras.callbacks.EarlyStopping(
            #     monitor = "val_loss",
            #     min_delta = 0.02,
            #     patience = 0,
            #     verbose = 1,
            #     mode = "auto",
            #     baseline = None,
            #     restore_best_weights = False,
            # )



            model = tf.keras.models.Sequential([
                 # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
                 # The first convolution
                 # СЛОИ СВЕРТКИ

                 # tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (target_size[0], target_size[1], 3)),   #64
                 # tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

                 # tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (1,1)),

                 #tf.keras.layers.Conv2D(16, (3,3), padding = 'same', activation = 'relu'),
                 #tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides = (1,1)),

                 #tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu'),
                 #tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides = (1,1)),

                 #tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu'),
                 #tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = (1,1)),

                 #tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu'),
                 #tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = (1,1)),

                 # ==========================================================================

                tf.keras.layers.Conv2D(8, (3,3), padding = 'same', activation = 'relu', input_shape = (target_size[0], target_size[1], 3)),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

                tf.keras.layers.Conv2D(16, (3,3), padding = 'same', activation = 'relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

                tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

                tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

                tf.keras.layers.Dropout(0.25),

                tf.keras.layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

                 # ========================================================================

                 # tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = (target_size[0], target_size[1], 3)),
                 # tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),

                 # tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                 # tf.keras.layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'),

                 # tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                 # tf.keras.layers.Conv2D(256, (3,3), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(256, (3,3), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(256, (3,3), padding = 'same', activation = 'relu'),

                 # tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                 # tf.keras.layers.Conv2D(512, (3,3), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(512, (3,3), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(512, (3,3), padding = 'same', activation = 'relu'),

                 # tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                 # tf.keras.layers.Conv2D(512, (3,3), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(512, (3,3), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(512, (3,3), padding = 'same', activation = 'relu'),

                 # tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                 # ========================================================================

                 # tf.keras.layers.Conv2D(96, (11,11), strides=(4,4), padding = 'same', activation = 'relu', input_shape = (target_size[0], target_size[1], 3)),
                 # tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
                 # tf.keras.layers.Conv2D(256, (5,5), strides=(1,1), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
                 # tf.keras.layers.Conv2D(384, (3,3), strides=(1,1), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(384, (3,3), strides=(1,1), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding = 'same', activation = 'relu'),
                 # tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),

                # СЛОЙ ПОДВЫБОРКИ
                tf.keras.layers.Dropout(0.5),
                # Flatten the results to feed into a dense layer
                tf.keras.layers.Flatten(),
                # 256 neuron in the fully-connected layer
                tf.keras.layers.Dense(4096, activation = 'relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation = 'relu'),
                # 5 output neurons for 5 classes with the softmax activation
                tf.keras.layers.Dense(num_classes, activation = 'softmax')
            ])

            # model = tf.keras.models.Sequential()
            # base_model = InceptionV3(weights='imagenet', include_top=False, classes=num_classes, input_shape=(target_size[0],target_size[1],3))
            # base_model.trainable = False
            # model.add(base_model)
            # model.add(tf.keras.layers.GlobalAveragePooling2D())
            # model.add(tf.keras.layers.Dropout(0.5))
            # model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
            # model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
            # model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

            plot_model(model, to_file = self.model_path + str(model_name)[:-3] + '.png')
            print("[INFO] ML model architecture had saved as image")

            if (summary): 
                #info = model.summary()
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                info = "\n".join(stringlist)


            model.compile(loss = 'categorical_crossentropy',
                          optimizer = Adam(lr = 0.001),
                          metrics = ['acc'])

            total_sample = train_generator.n
            valid_sample = valid_generator.n

            history = model.fit(
                    train_generator, 
                    steps_per_epoch = int(total_sample / batch_size),  
                    epochs = n_epochs,
                    verbose = verbose,
                    validation_data = valid_generator,
                    validation_steps = int(valid_sample / batch_size),
                    callbacks = [self.callback_3])
            
            #print(classification_report(val_datagen.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

            
            model.save(self.model_path + model_name)
            
            print("[INFO] Model training had finishd.")
            print("[INFO] " + model_name + " had saved in: ", self.model_path, " folder")
            
            #test_predictions = np.argmax(model.predict(test_ds), axis=-1)
            #print(classification_report(test['Cover_Type'],test_predictions))


            return 0, info

            #def test_torch_dekol_model(self, image):
            #    """
            #    Method for dekol model testing
            #    """
            #    with torch.no_grad():
            #        img = Image.fromarray(dekol_frame)
            #        inputs = self.preprocess(img).unsqueeze(0).to(self.device)
            #        outputs = self.model(inputs)
            #        _, preds = torch.max(outputs, 1)
            #        label = self.dekol_class_names[preds]

    def test_CNN(self, settings, target_size = (200, 200), info_flag = False):
        """
        Method for CNN model testing
        input    -> settings [json]
                 -> model_name [CNN classification model name]
                 -> target_size [size of matrix]
        input    -> 0
        """
        model_name = settings['NN']['keras_model_dekol_classification']
        
        if model_name is "":
            return 1

        else:
            folders_list = os.listdir(self.image_test_directory)
            folders_list.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            
            try:
                print("I am here!:" + str(self.model_path + model_name))
                model = load_model(self.model_path + model_name)
                if (model):
                    print(f"[INFO] Model loaded {model_name}")
                
                for fl in range(len(folders_list)):
                    images = fnmatch.filter(os.listdir(self.image_test_directory + str(folders_list[fl]) + "/"), '*.JPG')
                    # images = fnmatch.filter(os.listdir(self.image_test_directory + str(folders_list[fl]) + "/"), '*.jpg')
                    #print(len(images))
                    counter = 0
                    for i in tqdm(range(len(images))):
                        try:
                            image_default = Image.open(self.image_test_directory + str(folders_list[fl]) + "/" + images[i]).resize(target_size)
                            image_default = np.expand_dims(image_default, axis = 0)
                            image_default = (image_default-0.5)*2
                            
                            if info_flag == True: print(images[i], str(" -> "), np.around(model.predict(image_default)[0], decimals = 1))
                            # print("index: ",np.around(model.predict(image_default)[0], decimals = 1).index(1))
                            
                            # if(np.around(model.predict(image_default)[0], decimals = 1)[int(folders_list[fl])] == 1):
                            #     counter = counter + 1
                            prediction = list(model.predict(image_default)[0])
                            probability = max(prediction)
                            class_image = prediction.index(probability)
                            if class_image == int(folders_list[fl]):
                                counter += 1

                        except KeyboardInterrupt:
                            sys.exit(0)

                        except Exception as e: print(e)
                            

                    print("[INFO] Result persentage for " + str(folders_list[fl]) + ": ", round(counter / len(images) * 100, 2))
                    
            
            except Exception as e: 
                print("[INFO] ",e)
                print("[INFO] Early program termination!")
                return 1

            return 0
    
    def CNN(self, settings, image, model_name = '', target_size = (200, 200), info_flag = True):
        model = load_model(self.model_path + model_name)
        image_default = Image.fromarray(image).resize(target_size) 
        image_default = np.expand_dims(image_default, axis = 0)
                            
        #if info_flag == True: print(images[i], str(" -> "), np.around(model.predict(image_default)[0], decimals = 1))
        #if(np.around(model.predict(image_default)[0], decimals = 1)[int(folders_list[fl])] == 1):
        #    print("Done")
        return str(images[i], str(" -> "), np.around(model.c(image_default)[0], decimals = 1))

    def test_yolo_model_on_one_image(self, image, settings, nms_filter = True, thickness = 6, color = (0, 255, 255), fontScale = 2, font = cv.FONT_HERSHEY_SIMPLEX):
        """
        Method for weight file testing om single image
        input  -> image [RGB format], 
               -> settings [json], 
               -> nms_filter -> non_max_suppression algo
        output -> 
        """
        #class_names = {0: "bowl", 1: "sphera", 2: "pear_pot", 3: "pial", 4: "cyl_pot_20", 5: "cup", 6: "--", 7: "cyl_pot_inex", 8: "chay_10", 9: "kettle", 10: "2312", 11 : "erf", 12: "bhncvo"} #
        #class_names = {0:"cup", 1:"sph", 2:"kettle"}
        # for 29.03.21_1
        #class_names = {0:"pan_cell", 1:"pan_cell_obod", 2:"pan_cell", 3:"bowl", 4:"sphere", 5:"pan_cell_obod", 6:"cap",10:"tea_big",11:"tea_small" }
        # for 13.01.22
        class_names = {0:"bowl", 1:"sphere", 2:"pear_pot", 3:"pial", 4:"cyl_pot", 5:"cup", 7:"cyl_pot_20_inex", 8:"chay_10", 11:"kettle_35"} #pan_cell, pan_cell_obod, pan_cell_obod, tea_big
        try:
            LABELS_FILE = settings['NN']['vertical_view_labels_file']
            with open(LABELS_FILE, 'rt') as f:
                names = f.read().rstrip('\n').split('\n')
            
            net = cv.dnn_DetectionModel(settings['NN']['vertical_view_config_file'], settings['NN']['vertical_view_weights_file'])
            
            net.setInputSize(int(settings['NN']['input_size']), int(settings['NN']['input_size']))
            net.setInputScale(1.0 / 255)
            net.setInputSwapRB(True)
            
            classes, confidences, boxes = net.detect(image, confThreshold = float(settings['NN']['confidence_threshold']), nmsThreshold = float(settings['NN']['nms_threshold']))
            print("Classes wich was detected: ", boxes)
            
            if nms_filter == True:
                boxes = self.format_converter(boxes = boxes)
                #boxes = self.NMS(boxes = boxes, overlapThresh = 0.2)

                i = 0
                for box in zip(boxes):
                    #cropped_image = image[box[1]:(box[1] + box[3]), box[0]:int(box[0] + box[2])]
                    
                    #dekol = self.CNN(image = cropped_image, settings = settings, model_name = "model_v1_21.01.22.[12].h5", target_size = (200, 200), info_flag = True)
                    
                    cv.rectangle(image, (box[i][0], box[i][1]), (box[i][2], box[i][3]), color = (50, 250, 20), thickness = 8)
                    cv.putText(image, str(class_names[int(classes[i])]), (box[i][0], box[i][1]), font, fontScale, color, thickness, cv.LINE_AA, False)
                    #cv.putText(image, str(names[int(classId[i])]), (box[i][0], box[i][1]), font, fontScale, color, thickness, cv.LINE_AA, False)
                    #names[classId]
                    #cv.putText(image, str(dekol), (box[i][0], box[i][1] + box[i][3]), font, fontScale, color, thickness, cv.LINE_AA, False)
                    
                    #str(classes[i][0]), tuple(box[i][0], box[i][1])
                    i = i + 1
                
                cv.imwrite("/data/yolact/output/" + str(datetime.now()) + ".JPG", image)
                print(classes)
                return 0

            if nms_filter == False:
                for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                            
                    label = confidence
                    class_name = names[classId]
                    
                    #if (float(label) >= float(settings['NN']['confidence_threshold'])):
                    cv.rectangle(image, box, color = (50, 250, 20), thickness = 8)
                cv.imwrite("/data/darknet_AB/data/predict.JPG", image)
                return 0

        except Exception as e: 
            print(e)
            cv.imwrite("/data/yolact/output/" + str(datetime.now()) + ".JPG", image)
        #except:
        #    print(" ! Error in testing method. Check links for JPG, weight or cfg files")
        #    return 1


    def format_converter(self, boxes):
        """ 
        Format convertering method. 
        Convert x,y,w,h -> x1,y1,x2,y2
        input   -> boxes [x,y,w,h]
        output  -> np.array(boxes) [x1,y1,x2,y2]
        """
        data = []
        for i in range(len(boxes)):
            boxes[i][2] = boxes[i][2] + boxes[i][0]
            boxes[i][3] = boxes[i][3] + boxes[i][1] 
            data.append(tuple(boxes[i]))
        
        return np.array(data)


    def NMS(self, boxes, overlapThresh = 0.4):
        """ 
        Method for filtering boxes 
        input    -> boxes [boxes of detections], overlapThresh[treshold of boxes intersection degree]
        output   -> 'correct' box/boxes
        """
        if len(boxes) == 0:
            return []
        pick = []
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
            
        boxes = boxes[~np.all(boxes == 0, axis = 1)]

        idxs = np.argsort(y2)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]
            for pos in range(0, last):
                j = idxs[pos]
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                overlap = float(w * h) / area[j]
                if overlap > overlapThresh:
                    suppress.append(pos)
            idxs = np.delete(idxs, suppress)
        return boxes[pick]

class Drawing(object):
    
    def __init__(self):
        pass

    def draw_vertical_edges(self, image, left_x_edge, right_x_edge, color = (0, 255, 255), thickness  = 2):
        h,w = image.shape
        cv.line(image, (left_x_edge, 0) (left_x_edge, h), color, thickness)
        cv.line(image, (right_x_edge, 0) (right_x_edge, h), color, thickness)
        return image

    def draw_center_screen_divider (self, image, width, height, color, thickness):
        cv.line(image, (0, 0), (width, height), color, thickness)
        cv.line(image, (0, height), (width, 0), color, thickness)
        cv.arrowedLine(image, (int(width/10), int(height*.4)), (int(width/10), int(height*.6)), color, 3, 8, 0, 0.1)
        return image

    def draw_markers_labels(self, image, corners, ids = None, thickness = 6, color = (0, 255, 255), fontScale = 2, font = cv.FONT_HERSHEY_SIMPLEX):
        for i in range(len(corners)):
            cv.line(image, tuple(corners[i][0][0]), tuple(corners[i][0][2]), color, thickness)
            cv.line(image, tuple(corners[i][0][1]), tuple(corners[i][0][3]), color, thickness)
            if ids[i][0] is not None:
                cv.putText(image, str(ids[i][0]), tuple(corners[i][0][0]), font, fontScale, color, thickness, cv.LINE_AA, False)
            #    print(ids[0][i])
        return image    
   
    def draw_market_target(self, image, data, height, color, thickness):
        cv.circle(image, data, radius = int(height/8), color = color, thickness = thickness)
        #cv.circle(image, data, radius = int(height/20), color = color, thickness = thickness)
        return image

    def draw_center(self, image, x : int, y  : int, radius = 5 , color = (255, 255, 255), thickness = 2):
        cv.circle(image, (x, y), radius, color, thickness)
        return image

    def draw_label_name(self, image, text, place, font, fontScale, color, thickness):
        cv.putText(image, text, place, font, fontScale, color, thickness, cv.LINE_AA, False)
        return image

    def draw_vision_area(self, image, x, y):
        angle = 40
        startAngle = 0
        endAngle = 360
        color = (255, 0, 0)
        thickness = 3
        axesLength = (100, 50)
        cv.ellipse(image, (x, y), axesLength, angle, startAngle, endAngle, color, thickness)
        return image

class Video_Processing(object):
    
    def __init__(self):
        self.CHESSBOARD_CORNERS_ROWCOUNT = 9
        self.CHESSBOARD_CORNERS_COLCOUNT = 6
        self.calibration_images = glob.glob('../darknet_AB/python/src/calibration/*.bmp')
        self.calibration_result_folder = "../darknet_AB/python/src/calibration_result/"
        self.calibration_pckl = "../darknet_AB/python/src/calibration.pckl"
        self.calibration_yaml = "../darknet_AB/python/src/calibration.yaml"
        self.ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.ds_factor = 0.3


    def camera_calibration(self):
        
        objpoints = [] 
        imgpoints = [] 

        objp = np.zeros((self.CHESSBOARD_CORNERS_ROWCOUNT * self.CHESSBOARD_CORNERS_COLCOUNT,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.CHESSBOARD_CORNERS_ROWCOUNT,0:self.CHESSBOARD_CORNERS_COLCOUNT].T.reshape(-1, 2)

        print(" * Calibration on", len(self.calibration_images), "images is starting")
        
        imageSize = None 
        cnt = 0
        for iname in tqdm(self.calibration_images):

            img = cv.imread(iname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            board, corners = cv.findChessboardCorners(gray, (self.CHESSBOARD_CORNERS_ROWCOUNT,self.CHESSBOARD_CORNERS_COLCOUNT), None)

            if board == True:
                objpoints.append(objp)
                
                corners_acc = cv.cornerSubPix(
                        image = gray, corners = corners, winSize = (11, 11), zeroZone = (-1, -1),
                        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)) 

                imgpoints.append(corners_acc)

                if not imageSize:
                    imageSize = gray.shape[::-1]
            
                img = cv.drawChessboardCorners(img, (self.CHESSBOARD_CORNERS_ROWCOUNT, self.CHESSBOARD_CORNERS_COLCOUNT), corners_acc, board)
                cv.imwrite(self.calibration_result_folder + str(cnt) + str(".jpg"), img)
                cnt = cnt + 1
            
            else:

                print(" ! Not able to detect a chessboard in image: {}".format(iname))

        if len(self.calibration_images) < 1:
            print(" ! Calibration was unsuccessful. No images of chessboards were found. Add images of chessboards and use or alter the naming conventions used in this file.")
            exit()

        if not imageSize:
            print(" ! Calibration was unsuccessful. We couldn't detect chessboards in any of the images supplied. Try changing the patternSize passed into findChessboardCorners(), or try different pictures of chessboards.")
            exit()

        calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
                objectPoints = objpoints, imagePoints = imgpoints, imageSize = imageSize, cameraMatrix = None, distCoeffs = None)
            
        print("Camera matrix    :", cameraMatrix)
        print("Distortion coeff.:", distCoeffs)
            

        with open(self.calibration_pckl, "wb") as file:
            pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), file)
            #f.close()
        
        with open(self.calibration_yaml, "w") as file:
            yaml.dump((cameraMatrix, distCoeffs, rvecs, tvecs), file)

        print(' * Calibration successful. Calibration file used: {}'.format('calibration.pckl, calibration.yaml'))


    def find_color_card(self, image):
        """ Not used anymore"""
        arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = cv.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv.aruco.detectMarkers(image, arucoDict, parameters = arucoParams)

        try:
            ids = ids.flatten()

            i = np.squeeze(np.where(ids == 923))
            topLeft = np.squeeze(corners[i])[0]

            i = np.squeeze(np.where(ids == 1001))
            topRight = np.squeeze(corners[i])[1]

            i = np.squeeze(np.where(ids == 241))
            bottomRight = np.squeeze(corners[i])[2]

            i = np.squeeze(np.where(ids == 1007))                    
            bottomLeft = np.squeeze(corners[i])[3]
            print(topLeft, topRight, bottomRight, bottomLeft)

        except:
            return None
       
        cardCoords = np.array([topLeft, topRight,
            bottomRight, bottomLeft])
        
        card = four_point_transform(image, cardCoords)

        return card

    def dhash(self, image, hashSize = 8):
        """
        Method for image hashing
        INPUT   -> image in RGB format
        OUTPUT  -> hash sum
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hashSize + 1, hashSize))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    def changing_detection(self, image_previos, image_current):
        """
        Method for changing detection between previos and current images
        INPUT   -> image_previos and image_current in RGB format
        OUTPUT  -> status of changing TRUE or FALSE
        """
        
        hashes = {}
        images = [image_previos, image_current]
            
        for i in images:
            h = self.dhash(i)
            p = hashes.get(h, [])
            p.append(i)
            hashes[h] = p

        for (h, hashedPaths) in hashes.items():
            if len(hashedPaths) > 1:
                return True

            else:
                return False

    def add_weight(self, image, box, w1 = .5, w2 = .5):
        """ 
        Adding weights to th image method.
        input    -> image 
                 -> tuple (x1,y1,x2,y2)
        """
        output = image.copy()
        cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), -1)
        cv.addWeighted(image, w1, output, 1 - w2, 0, output)
        return output

    def color_correction(self, ref, image):
        """ 
        Not used anymore
        """
        #print(" * ")
        #print(" * Image loading")
        #ref = cv.imread(reference_path)
        #image = cv.imread(image_path)
        #ref = imutils.resize(ref, width = 600)
        #image = imutils.resize(image, width = 600)
        
        
        #print(" * Finding color matching cards")
        refCard = self.find_color_card(ref)
        #print("*", refCard)
        imageCard = self.find_color_card(image)
        #print("!", imageCard)
        if refCard is None or imageCard is None:
            #print(" * Could not find color matching card in refCard :c")
            sys.exit(0)

        else:
            print(" * Done!")

        imageCard = exposure.match_histograms(imageCard, refCard, 
            multichannel = True)


        #cv.imwrite("color_cerrection_image.JPG", imageCard)
        # show our input color matching card after histogram matching
        return imageCard


    def image_mask(image, lower, upper, total_area):
        try:
            hsv = cv.cvtColor(cv.GaussianBlur(image, (3, 3), 0), cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower, upper)
            mask = cv.erode(mask, None, iterations = 2)
            mask = cv.dilate(mask, None, iterations = 2)
            kernel = np.ones((5,5),np.uint8)
            edge_detected_image = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernel)
            contours, hierarchy = cv.findContours(edge_detected_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour_list = []
            area_list = []
            for contour in contours:
                approx = cv.approxPolyDP(contour,0.01 * cv.arcLength(contour, True), True)
                area = cv.contourArea(contour)
                if ((len(approx) > 7) & (area > 30) & (area < 0.1* total_area)):
                    contour_list.append(contour)
                    area_list.append(area)
            median_size = statistics.median(area_list)
            return contour_list, median_size

        except:
            return [], []
        

    def polka_detect(self, image):
        
        polka_size = ["myatnaya_polka", "goroh"], 
        polka_color = ["w", "r"]
        """Polka dekol detection method"""
        
        h, w, _ = image.shape
        total_area = w * h

        # white mask in HSV space
        whiteLower = (0, 0, 60)
        whiteUpper = (180, 60, 255)

        # red mask in HSV space
        redLower = (0, 150, 50)
        redUpper = (10, 255,255)
      
        contour_list, median_size = self.image_mask(image = image, lower = whiteLower, upper = whiteUpper, total_area = total_area)
        circle_counter = len(contour_list)

        # if circles are WHITE -> they are exists

        if (circle_counter > 1):
            
            if((circle_counter < 12) & (median_size > 0.01 * total_area)):
                return True, polka_size[1], polka_color[0]
            
            elif((circle_counter > 10) & (median_size < 0.01 * total_area)):
                return True, polka_size[0], polka_color[0]

        else:

            contour_list, median_size = self.image_mask(image = image, lower = redLower, upper = redUpper, total_area = total_area)
            circle_counter = len(contour_list)
            if (circle_counter > 1):
                if((circle_counter < 12) & (median_size > 0.01 * total_area)):
                    return True, polka_size[1], polka_color[1]

                elif((circle_counter > 10) & (median_size < 0.01 * total_area)):
                    return True, polka_size[0], polka_color[1]

                else:
                    return False, 0, 0
            else:
                return False, 0, 0   


    def marks_detection(self, checker, gray, cameraMatrix, distCoeffs):
        """
        """
        if (checker == 1):
            
            #board = aruco.GridBoard_create(
            #        markersX = 2, markersY = 2, markerLength = 0.045,
            #        markerSeparation = 0.01, dictionary = self.ARUCO_DICT)
     
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters = self.ARUCO_PARAMETERS)
                    
            #corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            #        image = gray, board = board, detectedCorners = corners, detectedIds = ids,
            #        rejectedCorners = rejectedImgPoints, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs)   

            #frame = aruco.drawDetectedMarkers(frame, corners, borderColor = (0, 0, 255))
            #print(corners)
            if type(corners) is "NoneType":
                print(" ! No marks")
                return 0

            elif (len(corners) > 0 and len(corners) <= 4):
                #print(corners)
                return corners, ids
                '''distance = self.dist_between_two_points(
                    x1 = corners[0][0][0][0], x2 = corners[1][0][0][0],
                    y1 = corners[0][0][0][1], y2 = corners[1][0][0][1])
                            
                    diameter_metric = real_distance_between_markers * diameter / distance'''
                    #corners[0][0][0][0], corners[0][0][0][1], corners[1][0][0][0], corners[1][0][0][1]


            elif (len(corners) == 0):
                #print(" ! Marks have not detected")
                return 0
                
            elif(len(corners) > 4):
                #print(" ! A lot of marks")
                return 0
            else:
                print(" ! Other problem with detection")
       
    def visual_object_orientation (self, image, corners = None, ids = None, color = (255, 255, 255), thickness = 6, fontScale = 2, font = cv.FONT_HERSHEY_SIMPLEX):
        height, width = image.shape[:2]
        circle_centers_coordinates = [(int(width/5), int(height*.8)),(int(width*.8), int(height*.8)),(int(width*.8), int(height/5)),(int(width/5), int(height/5))]
        
        image = Drawing().draw_center_screen_divider (image, width = width, height = height, color = color, thickness = thickness)

        if not (corners is None and ids is None):
            # По каждому центру
            for i in range(len(circle_centers_coordinates)):
                color = (255, 255, 255)
                image = Drawing().draw_label_name(image, text = str(i + 1), place = circle_centers_coordinates[i], font = font, fontScale = fontScale, color = color, thickness = thickness)
                new_ids = list(map(lambda x: x - 1, [j for i in ids for j in i]))

                if i in new_ids:
                    color_counter = 0
                    for j in range(4):
                        if(pow((corners[new_ids.index(i)][0][j][0] - circle_centers_coordinates[i][0]),2) + pow((corners[new_ids.index(i)][0][j][1] - circle_centers_coordinates[i][1]),2) <= pow(int(height/8),2)):
                            color_counter = color_counter + 1
                    
                    if (color_counter == 4):
                        color = (0, 255, 0)

                    elif (color_counter < 4):
                        color = (255, 255, 255)
                    

                    image = Drawing().draw_market_target(image, data = circle_centers_coordinates[i], height = height, color = color, thickness = thickness)
                    
                    color_counter = 0
                    
                    #if (len(new_ids) == 4):
                    #    self.xy2x_tranformation(*, *, *, *, camera_id = 1)

                else:
                    color = (255, 255, 255)
                    image = Drawing().draw_market_target(image, data = circle_centers_coordinates[i], height = height, color = color, thickness = thickness)

                
        return image 


    def object_size(self, gray, boxes):
        sizes = []        

        for i in range(len(boxes)):
            crop_gray_img = gray[int(boxes[i][1]*0.95):int(boxes[i][3]*1.05), int(boxes[i][0]*0.95):int(boxes[i][2]*1.05)]

            blur = cv.GaussianBlur(crop_gray_img, (5, 5), 0)

            threshed = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            threshed = (255-threshed)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

            #binary = cv.morphologyEx(threshed, cv.MORPH_GRADIENT, kernel, iterations = 1)

            binary = cv.morphologyEx(threshed, cv.MORPH_OPEN, kernel, iterations = 1)
            #binary = cv.dilate(binary,kernel,iterations = 15)
            cnts = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]
            #cv.drawContours(img, cnts, -1, (255, 0, 0), 2, cv.LINE_AA)

            for cnt in cnts:
                if cv.contourArea(cnt) < 5000 :
                    continue

                rbox = cv.fitEllipse(cnt)
                #cv.ellipse(img, rbox, (255, 100, 255), 2, cv.LINE_AA)
                sizes.append((rbox[0][0] + boxes[i][0]*0.95, rbox[0][1] + boxes[i][1]*0.95))

        return gray, centers


    def get_grayscale_frame(self, frame):
        #ret, frame = stream.read()
        frame = cv.resize(frame, None, fx = self.ds_factor,fy = self.ds_factor, interpolation = cv.INTER_AREA)
        #frame = cv2.resize(frame, (0,0), fx=stream_scale_factor, fy=stream_scale_factor)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return frame, gray 
 

    def adjust_gamma(self, image, gamma = 1.0):
        """Gamma correction method for image processing.  
        # input -> RGB image and  gamma parameter
        # output    -> image after corrections 
                -> table variable by looping over all pixel values in the range [0, 255]."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
            # apply gamma correction using the lookup table
        return cv.LUT(image, table)


    def motion_detection(self, current_gray, prvs_gray, hsv):
        """ Method for the motion detection. Can detect areas of motion extremums on the flow
        input -> gray frame in grayscale, prvs - frame from last step, hsv - one chennel in hsv format
        output -> contours and unused variable
        """
        flow = cv.calcOpticalFlowFarneback(
            prvs_gray, current_gray, None, pyr_scale = 0.5, levels = 5, 
            winsize = 15, iterations = 5, poly_n = 7, poly_sigma = 1.5, flags = 0)

        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2

        hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        h, s, v1 = cv.split(hsv)

        v1 = self.adjust_gamma(v1, 3)

        ret, v = cv.threshold(v1, 80,255, cv.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)

        v = cv.erode(v, kernel, iterations = 1)
        v = cv.dilate(v, kernel, iterations = 1)

        return cv.findContours(v.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], v

    def motion_indication(self, contour, image):
        M = cv.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x, y, w, h = cv.boundingRect(contour)
        rx = x + int(w / 2)
        ry = y + int(h / 2)
        ca = cv.contourArea(contour)
        #cv2.drawContours(frame_res,[c],0,(0,255,0),2)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 8)
        return w, h

    def dekol_place_detection(self, frame, 
        lower_level_mask = np.array([0,0,0]), 
        upper_level_mask = np.array([180,70,255]), 
        kernel = np.ones((3, 3), np.uint8),
        crop_index = 0.2):

        h_img, w_img, _ = frame.shape
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # chose place of mask    
        total_image_minus_color_mask = cv.inRange(hsv, lower_level_mask, upper_level_mask)

        # mask filtering
        total_image_minus_color_mask = cv.erode(total_image_minus_color_mask,kernel,iterations = 1)
        # cut small areas around the main mask
        total_image_minus_color_mask = cv.morphologyEx(total_image_minus_color_mask, cv.MORPH_OPEN, kernel, iterations = 1) 
        total_image_minus_color_mask = cv.dilate(total_image_minus_color_mask,kernel,iterations = 2)

        # get mask of object by it color
        contours, _ = cv.findContours(total_image_minus_color_mask.copy(), 1, 1) 

        if len(contours) != 0:

            hull = cv.convexHull(max(contours, key = cv.contourArea))
            img_copy = frame.copy()
            #cv.drawContours(img_copy, contours = [hull], contourIdx = 0, color = (255, 255, 0), thickness = 2)
            interpolate_bowl_mask = cv.drawContours(np.zeros_like(total_image_minus_color_mask), [hull], -1, (255, 255, 255), cv.FILLED)

            # inverting
            internal_mask = cv.bitwise_not(total_image_minus_color_mask)

            # mask of bowl and dekol areas intersection
            bowl_and_dekol_intersection = cv.bitwise_and(interpolate_bowl_mask, internal_mask, mask = None)
            h_intersection, w_intersection = bowl_and_dekol_intersection.shape
            # cropping territory with predicted dekol 
            control_mask = cv.rectangle(np.zeros(bowl_and_dekol_intersection.shape[:2], dtype="uint8"), 
                (int(w_intersection * crop_index), int(h_intersection * crop_index)), 
                (int(w_intersection * (1 - crop_index)), int(h_intersection * ( 1 - crop_index))), 255, -1) 
            
            dest_and_2 = cv.bitwise_and(bowl_and_dekol_intersection, control_mask, mask = None)
            
            cnts, _ = cv.findContours(dest_and_2.copy(), 1, 1)
            x,y,w,h = cv.boundingRect(max(cnts, key = cv.contourArea))

            return frame[y:y + w, x:x + h]

    def inner_color_detection(self, image, boxes):
        """ 
        Method for inner color calculation.
        """
        colors_list = []
        if image:
            image_buf = image.copy()
            h, w, _ = image_buf.shape
            for i in range(len(boxes)):
                image_object = image_buf[int((boxes[i][1] - boxes[i][3] / 2) * h):int((boxes[i][1] + boxes[i][3] / 2) * h), int((boxes[i][0] - boxes[i][2] / 2) * w):int((boxes[i][0] + boxes[i][2] / 2) * w)]
                predicted_color = self.get_area_by_color_mask(image = image_object) 
                colors_list.append(predicted_color)
        
        return colors_list
    
    def get_area_by_color_mask(self, image):

        colors = {
            'belaya': [np.array([0, 0, 125]), np.array([180, 25, 250])],
            'valentina': [np.array([105, 100, 1]), np.array([130, 255, 255])],
            'begevaya': [np.array([0, 35, 1]), np.array([30, 100, 250])],
            'krasnaya' : [np.array([165, 100, 1]), np.array([180, 255, 250])],
            'krasnaya_c' : [np.array([0, 100, 1]), np.array([10, 255, 250])],
            'begevaya_c': [np.array([150, 35, 1]), np.array([180, 100, 250])],
            'svetlo-salat' : [np.array([40, 25, 1]), np.array([90, 100, 250])]
            }


        list_of_colors = list(colors.keys())
        list_of_weights = []
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        # BLUE
        #lower_red = np.array([50, 1, 1])
        #upper_red = np.array([150, 255, 255])

        # OHRA
        #lower_red = np.array([1, 1, 1])
        #upper_red = np.array([50, 360, 200])
        
        #WHITE
        #lower_red = np.array([0, 1, 150])
        #upper_red = np.array([360, 360, 360])
        
        for i in range(len(list_of_colors)):
            #print(colors[list_of_colors[i]][0], colors[list_of_colors[i]][1])
            mask = cv.inRange(hsv, colors[list_of_colors[i]][0], colors[list_of_colors[i]][1])
            res = cv.bitwise_and(image, image, mask = mask)
            
            try:
                pixels = cv.countNonZero(mask)
                image_area = image.shape[0] * image.shape[1]
                area_ratio = (pixels / image_area) * 100
                #print("Have got area_ratio= ", area_ratio)
            
            except:
                #print("Any ratio")
                area_ratio = 0
            
            list_of_weights.append(area_ratio)

        max_predict_level = max(list_of_weights)
        predicted_color_index = list_of_weights.index(max_predict_level)
        predicted_color = list_of_colors[predicted_color_index]

        '''
        
        ########## OLD ####################################################
        #mask = cv.inRange(hsv, lower_red, upper_red)
        #res = cv.bitwise_and(image, image, mask = mask)
        #try:
        #    pixels = cv.countNonZero(mask)
        #    image_area = image.shape[0] * image.shape[1]
        #    area_ratio = (pixels / image_area) * 100
        #    return round(area_ratio,1), res
        
        #except:
        #    return 0, mask
        '''
        predicted_color = list_of_colors[predicted_color_index]
        #print(predicted_color, max_predict_level)
        

        return predicted_color
    
    def file_checker(self):
        """
        Checking exists of camera calibration file 
        INPUT     -> 
        OUTPUT    -> status, cameraMatrix, distCoeffs
        """
        if not os.path.exists(self.calibration_pckl):
            print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
            #exit()
            return 0, 0, 0

        else:
            f = open(self.calibration_pckl, 'rb')
            (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
            f.close()
            if cameraMatrix is None or distCoeffs is None:
                print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
                exit()
                return 0, 0, 0

            return 1, cameraMatrix, distCoeffs

    def circle_detection(self, frame, gray, minDist = 200, param1 = 30, param2 = 60, minRadius = 80, maxRadius = 350, blurred_value = 25, thickness = 25):
        """
        Method for circle detection. Help to find plates from vertical camera
        input  -> 
        output -> rgb_frame, centers, radiuses
        """ 
        centers = []
        radiuses = []          
        blurred = cv.medianBlur(gray, blurred_value)
        circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv.circle(frame, (i[0], i[1]), i[2], (255,153,204), thickness)
                centers.append((i[0], i[1]))
                radiuses.append(round(i[2] * 0.1059,1))
                #cv.putText(frame, str(round(i[2] * 0.1059,1)) + str(" cm"), (i[0], i[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 6, cv.LINE_AA, False)
        return frame, centers, radiuses

class BarCode(object):
    
    def __init__(self):
        pass

    def Recognition(self, image):
        detectedBarcodes = decode(image)
        for barcode in detectedBarcodes:
            (x, y, w, h) = barcode.rect
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
            print(x, y, x + w, y + h)
            print(barcode.data)
            print(barcode.type)

        return image




    
