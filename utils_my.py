#!/usr/bin python3

__title__ = "Utils"
__version__ = "1.1"
__license__ = "None"
__copyright__ = "None"
__category__  = "Library"
__status__ = "Development"
__author__ = "Ivan Zhezhera"
__company__ = "it-enterprise"
__maintainer__ = "None"
__email__ = "zhezhera@it-enterprise.com"

# import the necessary packages
from imutils.perspective import four_point_transform
from skimage import exposure
import skimage.io 
import skimage.segmentation
from skimage.segmentation import mark_boundaries
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression

import os, sys, glob
from os import walk
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imutils
from imutils import paths

import shutil
import cv2 as cv
import moviepy.video.io.ImageSequenceClip
import fnmatch
import random
import math
import copy
import warnings

from tqdm import tqdm
from datetime import datetime
import time

from pyzbar import pyzbar
from pytesseract import *
from PIL import Image

import torch
from torchvision import models, transforms
from pathlib import Path
import subprocess
import smtplib, ssl, email
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

import video_processing as vp
import camera as cm

import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import InteractiveSession, ConfigProto
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications.imagenet_utils import decode_predictions

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from lime import lime_image


#config_file =  #yolov4-tiny-custom_light_4.cfg" #config.cfg" 
#weights_file = #"/data/darknet_AB/python/src/29.03.21_1.weights" #01.06.21.weights" #
#segmen_files_path = '/home/iiot/source/segment_2/data/json'

def keras_model_creation(path, model_name, target_size, info_flag):
    result = vp.Net().keras_model_creation(path = path, model_name = model_name, target_size = target_size, info_flag = info_flag)
    if result == 0:
        print("[INFO] Model " + str(model_name) + " has done!")
    else:
        print("[INFO] Error in model " + str(model_name) +" creation")

def keras_stack_models_training(models_path, images_path, batch_size, n_epochs, verbose, target_size, info_flag):
    result = vp.Net().stack_keras_models_training(
        models_path = models_path, 
        images_path = images_path, 
        batch_size = batch_size, 
        target_size = target_size, 
        n_epochs = n_epochs, 
        verbose = verbose, 
        info_flag = info_flag)

    if result == 0:
        print("[INFO] Models stack has done!")

    else:
        print("[INFO] Some error with models stack")

def system_version():
    return data

def recurcive_folders_cleaning(path, file_format):
    try:
        for parent, dirnames, filenames in os.walk(path):
            for fn in filenames:
                if fn.lower().endswith(file_format):
                    os.remove(os.path.join(parent, fn))
        print("[INFO] Files have deleted!")

    except:
        print('[INFO] Uncorrect input data')

def folders_cleaning(train_path, valid_path): #train_path, valid_path
    """
    Folder preparation method. Used for preliminary cleaning of training and 
    test data storage folders (of iamages and labels).
    input  -> train_path and valid_path (train and valid path's)
    otput  -> 0
    """
    #os.chdir(self.parent_path)
    list_of_address = [train_path, valid_path]
    for i in range(len(list_of_address)):
        files = glob.glob(list_of_address[i] + str('*'))
        for f in files:
            os.remove(f)
    print("[INFO] Cleaning of all folders has done")
    return 0

def recursive_total_cleaning(path: str):

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            print(' ! Failed to delete %s. Reason: %s' % (file_path, e))

def data_preparation(settings):
    """ 
    Data preparation method. Used for starting file spliating and 
    file preparation methods.
    """
    path = settings['links']['images_and_labels']
    docker_images_train_path = "data/train"
    docker_images_valid_path = "data/valid"
    file_path = "../darknet_AB/data/"

    train_file = settings['links']['train_file']
    valid_file = settings['links']['valid_file']
    train_data_path = settings['links']['train_data_path']
    vaild_data_path = settings['links']['valid_data_path']

    print(' * Data preparation in process...')
    print(' * Cheaking quantity of images and labels')
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    labels = fnmatch.filter(os.listdir(path), '*.txt')
    print("Images: ", len(images)) 
    print("Labels: ", len(labels))
    if(len(images) == len(labels)):
        split_process(images = images, src_path = path, train_data_path = train_data_path, vaild_data_path = vaild_data_path)

    elif (len(images) > len(labels)):
        new_images = []
        for i in range(len(labels)):
            new_images.append(labels[i].replace("txt", "JPG"))

        split_process(images = new_images, src_path = path, train_data_path = train_data_path, vaild_data_path = vaild_data_path)
    open(str(file_path) + str (train_file), 'w').close()
    open(str(file_path) + str (valid_file), 'w').close()
    file_list_preparation(train_file, train_data_path, docker_images_train_path, file_path)
    file_list_preparation(valid_file, vaild_data_path, docker_images_valid_path, file_path)
    print("[INFO] Dataset ready for the model training! ")
    return 0

def lime_analysis(settings, image = "", model = "", report = True):
    """
    Method for RCNN model analysis. 
    Can make checking reaction of model on image and change color of used superpixels.
    input  -> iamge [RGB format]
           -> model [*.h5 - keras format file]
           -> report [boolean flag for info printing]
    output
    """
    #https://www.machinelearningmastery.ru/interpretable-machine-learning-for-image-classification-with-lime-ea947e82ca13/
    np.random.seed(222)

    #warnings.filterwarnings('ignore') 
    #inceptionV3_model = keras.applications.inception_v3.InceptionV3() #Load pretrained model
    # LAST: model_v1_20.01.22.[12].h5
    model = load_model(model) #"/data/yolact/tf_models/model_v1_29.10.21.[5].h5"
    print("[INFO] Model " + str(model) + " has downloaded")

    Xi = skimage.io.imread(image) #'/data/yolact/DEKOL/classification/imds_nano_3/data_from_progon/2/2021-11-26 00:10:04.702434.JPG'
    Xi = skimage.transform.resize(Xi, (200,200)) 
    Xi = (Xi - 0.5)*2 #Inception pre-processing
    #skimage.io.imshow(Xi/2+0.5) # Show image before inception preprocessing

    # PREDICTION

    superpixels = skimage.segmentation.quickshift(Xi, 
        kernel_size = 2, 
        max_dist = 100, 
        ratio = 0.2)

    num_superpixels = np.unique(superpixels).shape[0]
    # num_superpixels

    #skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))
    #skimage.io.skimage.io..imsave('/data/yolact/LIME_RESULT.png', skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))
    # skimage.io.imsave('/data/yolact/DEKOL/classification/imds_small_four_classes/3_RESULT.png', skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))


    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(Xi, model.predict, hide_color=0)

    result = model.predict(np.expand_dims(Xi, axis = 0))
    print(f"\n[INFO] Result of prediction: {result[0]}\n")

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    skimage.io.imsave('/data/yolact/DEKOL/classification/imds_nano_9/RESULt_5.png', mark_boundaries(temp / 2 + 0.5, mask))

    # num_perturb = 150
    # perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    # print(perturbations) #Show example of perturbation

    # def perturb_image(img,perturbation,segments):
    #     active_pixels = np.where(perturbation == 1)[0]
    #     mask = np.zeros(segments.shape)
    #     for active in active_pixels:
    #         mask[segments == active] = 1 
    #     perturbed_image = copy.deepcopy(img)
    #     perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    #     return perturbed_image

    # skimage.io.imsave('/data/yolact/DEKOL/classification/imds_small_four_classes/3_RESULT_2.png', perturb_image(Xi/2+0.5,perturbations[0],superpixels))

    # print("[INFO] Image with segments has created")

    # predictions = []
    # for pert in perturbations:
    #     perturbed_img = perturb_image(Xi,pert,superpixels)
    #     pred = model.predict(perturbed_img[np.newaxis,:,:,:])
    #     predictions.append(pred)

    # predictions = np.array(predictions)
    # print(predictions.shape)


    # original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
    # distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
    # print(distances.shape)

    # original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
    # distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
    # print(distances.shape)

    # kernel_width = 0.25
    # weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
    # print(weights.shape)

    # top_pred_classes = np.array([0,1,2,3,4])

    # class_to_explain = top_pred_classes[0]
    # simpler_model = LinearRegression()
    # simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
    # coeff = simpler_model.coef_[0]
    # print("======================")
    # print(coeff)
    # print("======================")

    # num_top_features = 4
    # top_features = np.argsort(coeff)[-num_top_features:] 
    # print(top_features)

    # mask = np.zeros(num_superpixels) 
    # mask[top_features]= True #Activate top superpixels
    # skimage.io.imsave('/data/yolact/DEKOL/classification/imds_small_four_classes/3_RESULT_3.png',perturb_image(Xi/2+0.5,mask,superpixels) )
    # print("[INFO] Last step is Done!")

    # print("[INFO] Report is done!")

def CNN_dekol_model_training(settings, n_epochs: int, model_name : str):
    result = 1
    result = vp.Net().train_CNN(model_name = model_name, n_epochs = n_epochs, settings = settings)
    return result

def CNN_dekol_model_testing(settings):
    result = vp.Net().test_CNN(settings = settings)
    if result == 0: 
        print("[INFO] Done!")
    else:
        print("[INFO] Finished with error.")

def colored_clone(path = './../darknet_AB/data/total_data/'):
    #image, image_path, label, label_path
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    labels = fnmatch.filter(os.listdir(path), '*.txt')
    #h,w,channels = cv.imread(path + images[0]).shape
    #print(h,w,channels)
    for i in tqdm(range(len(images))):
        image = cv.imread(path + images[i])
        #(B, G, R) = cv.split(image)
        #image_copy = image.copy()
        #image_copy = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)]
        
        # h, w, channels = image_copy.shape
        #image_copy = np.ones((h,w,3))

        #image = cv.add(image, image_copy, dtype=cv.CV_64F)
        #print(h,w,channels)
        #image_copy = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)]
        #image_copy = (1, 190, 200)
        #image = cv.absdiff(image, image_copy)
        for j in range(3):
            #part = random.uniform(0.1, 0.5)
            #image = cv.addWeighted(image, part, image_copy, 1 - part, 0,dtype = cv.CV_64F)
            #image[:, :, 0] = (image[:, :, 0] * random.uniform(0.1,255)).clip(0, 1)
            #image = cv.merge([G, R, B])
            #cv.merge([zeros, zeros, R])
            image_new = vp.Image_augmentation().image_noise(image)
            image_new = vp.Image_augmentation().image_gamma_correction(image = image_new, gamma = random.randint(1, 5))
            image_new = vp.Image_augmentation().image_blur(image = image_new, k_param = random.randint(1, 5))

            cv.imwrite(path + str(images[i]).replace(".JPG", "_" + str(j) + "_clone.JPG"), image_new)
            shutil.copy(path + str(labels[i]), path + str(labels[i]).replace(".txt", "_" + str(j) + "_clone.txt"))

def image_data_build_up_light(path = ''):
    """ 
    Function need just for the image data build-up for classification task
    """

    images = fnmatch.filter(os.listdir(path), '*.JPG')
    for i in tqdm(range(len(images))):
        image = cv.imread(path + images[i])
        image_new = vp.Image_augmentation().flip_horizontal(image)
        cv.imwrite(path + str(images[i]).replace(".JPG", "_" + str(i) + "_clone.JPG"), image_new)

def add_color_mask(path = ''):
    """
    Method for visual checking of labeling quality
    """
    counter = 0
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    labels = fnmatch.filter(os.listdir(path), '*.txt')

    for i in tqdm(range(len(images))):
        try:
            #if(images[i].replace("JPG","txt") in labels):
            with open(path + images[i].split(".")[0] + '.txt', 'r') as file:
                image = cv.imread(path + images[i])
                x_center = 0; y_center = 0; w = 0; h = 0
                data = file.read().replace('\n', '')
                x_center_relative = float(data.split(" ")[1])
                y_center_relative = float(data.split(" ")[2])
                w_relative = float(data.split(" ")[3])
                h_relative = float(data.split(" ")[4])
                h,w,_ = image.shape
                #print(h,w)
                if (x_center_relative > 0 and y_center_relative > 0):
                    box = (int((x_center_relative - w_relative / 2) * w), int((y_center_relative - h_relative / 2) * h), int((x_center_relative + w_relative / 2) * w), int((y_center_relative + h_relative / 2) * h))
                    image = vp.Video_Processing().add_weight(image = image, box = box)
                    cv.imwrite(path + str("copy_") + images[i], image)
        except:
            print("[INFO] Some trable with file: ", images[i], " or it label file", )
            
    persent = counter / len(images) * 100
    print("[INFO] Process is Done! \n[INFO] Was ",persent, "% of errors")

def split_process(images, src_path, train_data_path, vaild_data_path, split_persentage = 0.85):
    """ Split process. Can split image data for two parts: test and 
    train by split_persentage and also creat list of label files. 
    input  -> images, list of images names
              split_persentage = 0.9
    output -> 0"""        
    image_train_data = random.sample(images, k = round(split_persentage * len(images)))

    file_mover(list(image_train_data), src_path, train_data_path)

    valid_data = list(set(images) ^ set(image_train_data))

    file_mover(valid_data, src_path, vaild_data_path)
    # test and train labels lists creation 
    label_train_data = []
    for string in image_train_data:
        label_train_data.append(string.replace("JPG", "txt"))

    file_mover(list(label_train_data), src_path, train_data_path)
    label_valid_data = []
    for string in valid_data:
        label_valid_data.append(string.replace("JPG", "txt"))

    file_mover(label_valid_data, src_path, vaild_data_path)
    return 0

def file_mover(files, path_from : str, path_to : str):
    """ Can make files movement from path to path
    input  -> files, list of files name 
              path_from, string path to start folder
              path_to, string path to end folder
    output -> 0
    """
    for i in range(len(files)):
        if(files[i] != "backgraund.JPG"):
            shutil.move(path_from + files[i], path_to + files[i])
    return 0

def file_list_preparation(file_name : str, images_path : str, docker_images_path : str, file_path : str): 
    """Train and valid list's creation method"""
    _, _, filenames = next(walk(images_path))

    with open(str(file_path) + str (file_name), "r+") as file_object:

        file_object.seek(0)

        data = file_object.read(200)
        if len(data) > 0 :
            file_object.write("\n")

        for i in range(len(filenames)):
            if os.path.splitext(filenames[i])[1] == ".JPG":

                filenames[i] = str(docker_images_path) + str("/") + str(filenames[i]) + str('\n')
                file_object.write(filenames[i])

    print("[INFO] ", file_name, "has created")
    return 0   


    '''def segmentation_labels_creation():
        #####################################################################################################
        # NOT USED ANYMORE
        # It was supposed to segment the data for further auto-marking of the general view of the dishes. 
        #####################################################################################################
        def sublist_in_list(data, key):
            return [s for s in data if key in s]

        def lebels_creator():
            _, _, filenames = next(walk(segmen_files_path))
            ss = sublist_in_list(data = filenames, key = ".json")
            os.chdir("train_segmentation")
            for i in tqdm(range(len(ss))):
                os.system(str("labelme_json_to_dataset ") + str(ss[i]) + str(" -o ") + str(ss[i].replace(".","_")))

        def png_catcher():
            os.chdir("train_segmentation")
            list_dir = next(os.walk('.'))[1]
            list_dir.remove("annotations")

            for i in tqdm(range(len(list_dir))):
                shutil.move(str(list_dir[i]) + str("/label.png"), str("annotations/")+str(list_dir[i][:-5])+str(".png"))

        png_catcher()
        #labels_creator()'''

def fast_video_to_frames(video_path : str, rotation, settings):
    image_folder =  settings['links']['screenshots']
    vidcap = cv.VideoCapture(video_path)
    def getFrame(sec):
        vidcap.set(cv.CAP_PROP_POS_MSEC,sec * 1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv.imwrite(image_folder + str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames

    sec = 0
    frameRate = 0.18 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
    print("[INFO] Done!")

def model_testing(settings):
    """
    Model testing runner
    """
    path = settings['links']['input_data_path']

    if path:
        images = fnmatch.filter(os.listdir(path), '*.JPG')
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        #print(images)
        #images.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        for i in tqdm(range(len(images))):
            image = cv.imread(path + images[i])
            vp.Net().test_yolo_model_on_one_image(image = image, settings = settings)

        print("[INFO] Done!")

    else:
        print("[INFO] Uncorrect path.")

def video_writer_posuda_horizontal(video_path : str, rotation : bool, settings):
    #print("Starting to process video: ", video_path)
    #>>>>>>>>>>>Work jist unner the docker<<<<<<<<<<<<<<"
    #input_video = cv.VideoCapture(video_path)
    #fourcc = cv.VideoWriter_fourcc(*settings['video']['codec'])
    #width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    #height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    #size = (width, height)
    #total_frames = input_video.get(cv.CAP_PROP_FRAME_COUNT)
    #c = 0
    #image_folder =  settings['links']['screenshots']
    
    

    
    # Dekol model loading
    model = torch.load(settings['NN']['model_dekol'])
    model.eval()

    class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '2', '3', '4', '5', '6', '7', '8', '9'] 

    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(size = 256), #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        transforms.CenterCrop(size = 224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #train_path = '/data/yolact/DEKOL/classification/imds_small/train'
    #root = pathlib.Path(train_path)
    #classes_= sorted([j.name.split('/')[-1] for j in root.iterdir()])


    with open(settings['NN']['vertical_view_labels_file'], "rt") as f:
        names = f.read().rstrip("\n").split("\n")
    
    
    if input_video.isOpened():

        # for the numbers
        #numbers = {}
        #message = []

        print("[INFO] Frames making...")

        for i in tqdm(range(int(total_frames - 1))):
            try:
                if (c < total_frames):
                    ret, frame = input_video.read()
                    if (rotation == 1):
                        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
                    err_count = 0
                    #classes, confidences, boxes = net.detect(frame, confThreshold = CONFIDENCE_THRESHOLD, nmsThreshold = 0.4)

                    # ТОЛЬКО ДЛЯ ДЕМКИ УГЛОВОГО ВИДЕО!!!!
                    #hhh, www, _ = frame.shape
                    #frame = frame[int(hhh*.5):hhh, int(www*.2):int(www*.9)]
                    try:
                        classes, confidences, boxes = net.detect(frame, confThreshold = float(settings['NN']['confidence_threshold']), nmsThreshold = float(settings['NN']['nms_threshold']))
                    except:
                        pass
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    #centers = []
                    #*****************************************************************

                    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                        label = '%.2f' % confidence
                        print(classId, confidence, box)
                        if (float(label) >= float(settings['NN']['confidence_threshold'])):
                            dekol_name = 'None'
                            #label = '%s: %s' % (names[classId], label)
                            label = names[classId]

                            #print(names[classId], label, c)

                            # variant for the numbers
                            #label = '%s' % (names[classId])
                            
                            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            left_main, top_main, width_main, height_main = box

                            #print(width_main, height_main)

                            top_main = max(top_main, labelSize[1])
                            #print(top, frame.shape[0] * 0.4)

                            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            #centers.append([left_main + int(width_main/2), top_main + int(height_main/2)])
                            #*****************************************************************
                            
                            cropped_frame = frame[int(top_main + 0.4 * height_main):top_main + height_main, left_main:left_main + width_main]
                            cropped_frame = cropped_frame[int(0.33* cropped_frame.shape[0]):cropped_frame.shape[0], 0 : cropped_frame.shape[1]]
                            color_value = vp.Video_Processing().get_area_by_color_mask(image = cropped_frame)
                            
                            #x_offset = y_offset = 50
                            #frame[y_offset:y_offset+res.shape[0], x_offset:x_offset+res.shape[1]] = res  
                            

                            if (top_main > frame.shape[0] * 0.1):
                                # Dekol classification ########################################
                                top = int(top_main + 0.35 * height_main)
                                left = int(left_main + 0.2 * width_main)
                                height = int(top + 0.6 * height_main)
                                width = int(left + 0.6 * width_main)
                                
                                # УБИРАЕМ ВСЕ ЧАШКИ :(

                                if (width_main * height_main > 50000):
                                    # red box for dekol
                                    frame = cv.rectangle(frame, (left, top), (width, height), color = (0, 0, 255), thickness = 3)

                                    crop_img = frame[top:height, left:width]
                                    
                                    crop_img = vp.Image_augmentation().fill(image = crop_img, h = 256, w = 256)
                                    
                                    #crop_img = cv.resize(crop_img, (256,256))

                                    #cv.imwrite(str("./logs/") + str(datetime.now().strftime("%H:%M:%S")) + str(".JPG"), crop_img)
                                    
                                    crop_img = vp.Video_Processing().adjust_gamma(crop_img, gamma = 3.0)

                                    im_pil = Image.fromarray(crop_img)
                                    inputs = preprocess(im_pil).unsqueeze(0).to(device)
                                    outputs = model(inputs)
                                    _, preds = torch.max(outputs, 1)    
                                    dekol_name = class_names[preds]
                                    
                                
                                    # total object frame
                                    cv.rectangle(frame, box, color = (50, 250, 20), thickness = 4)
                                    
                                    # white backgraund for the text
                                    cv.rectangle(frame, (left_main, top_main), (int(left + width_main * 0.8), top_main + 100), (255, 255, 255), cv.FILLED)

                                    # label name
                                    cv.putText(frame, "LBL:" + label, (left_main, top_main + 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                                    # dekol name
                                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                    
                                    if(label.split(":")[0] == "Z01610710_CH_31A00"):
                                        dekol_name = "Polka"

                                    elif(label.split(":")[0] == "Z22711A6M237F00"):
                                        dekol_name = "Violetta"

                                    elif(label.split(":")[0] == "Z01612301!31S01"):
                                        dekol_name = "Zelen_Fresh"

                                    elif(label.split(":")[0] == "Z01911301!31S00"):
                                        dekol_name = "Zelen_Fresh"

                                    elif(label.split(":")[0] == "Z50308_L_14/00001" and color_value == 'L_GREEN'):
                                        dekol_name = "Fialka"

                                    elif(label.split(":")[0] == "Z50308_L_14/00001" and color_value == 'WHITE'):
                                        dekol_name = "Polka"
                                    else:
                                        dekol_name = "None"
                                    #*****************************************************************


                                    # БЫЛО 60 вместо 35
                                    cv.putText(frame, "DKL:" + dekol_name, (left_main, top_main + 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                                   
                                    # color name
                                    #if (color_value > 0):
                                    #color_value = color_value
                                    #cv.putText(frame, "CLR:" + color_value, (left_main, top_main + 85), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)                                
                                    cv.rectangle(frame, (left_main, top_main + 70), (int(left + width_main * 0.8), top_main + 100), (0, 0, 0), 3) 
                                    color_markers = {'WHITE': [255,255,255], 'VALENTINA': [128,0,0], 'L_GREEN': [181,228,255]}  
                                    rect_color = color_markers[color_value]
                                    cv.rectangle(frame, (left_main, top_main + 70), (int(left + width_main * 0.8), top_main + 100), rect_color, cv.FILLED)

                                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                                    
                                    if (label.split(":")[0] == "Z01610710_CH_31A00"):
                                        cv.rectangle(frame, (left_main, top_main + 70), (int(left + width_main * 0.8), top_main + 100), (128,0,0), cv.FILLED)
                                    #*****************************************************************'''


                                    
                                    #else:
                                    #    cv.putText(frame, "CLR:" + str("None"), (left, top + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))                                

                                    #numbers[left] = label
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!          
                    #print(centers)
                    #*****************************************************************

                    # sorting numbers by y coordinate
                    #for z in sorted (numbers.keys()) : 
                    #    message.extend(numbers[z])
                    
                    # variant for the numbers 

                    #cv.rectangle(frame, (0, 70), (1280 , 110), (255, 255, 255), cv.FILLED)
                    #if(len(message) == 12):
                    #    cv.putText(frame , str("ID: ") + str(message), (100 , 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                    #else:
                    #    cv.putText(frame , str("ID: "), (100 , 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)                    
                    
                    cv.imwrite(str(image_folder) + str(c) + str(".JPG"), frame)
                    c = c + 1
                    #message.clear()
                    #numbers.clear()


            except KeyboardInterrupt:
                input_video.release()
                break


            except:
                #cv.rectangle(frame, (0, 70), (1280 , 110), (255, 255, 255), cv.FILLED)
                #cv.putText(frame , str("ID: "), (100 , 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                cv.imwrite(str(image_folder) + str(c) + str(".JPG"), frame)
                c = c + 1

                if (c >= total_frames):
                    input_video.release()
                    print("[INFO] Starting to merge frames")

def deep_processing(frame, encoder, tracker, infer):
    """
    Method for deep processing tracking
    """
    if(1):   
        nms_max_overlap = 1.0
        
        image_data = cv.resize(frame, (416,416)) #(416, 416)
        image_data = image_data / 255.
        
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        
  
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores = tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class = 50,
            max_total_size = 25,
            iou_threshold = 0.3,
            score_threshold = 0.5
        )
        
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]
        
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        pred_bbox = [bboxes, scores, classes, num_objects]
        
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        allowed_classes = list(class_names.values())
        names = []
        deleted_indx = []

        
        
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]

            if class_name not in allowed_classes:
                deleted_indx.append(i)

            else:
                names.append(class_name)

        names = np.array(names)
        count = len(names)

        bboxes = np.delete(bboxes, deleted_indx, axis = 0)
        scores = np.delete(scores, deleted_indx, axis = 0)

        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)
        boxes = []
        track_ids = []
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            #cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            #cv.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            #cv.putText(frame, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255), 2)
            #print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
            track_ids.append(track.track_id)
            boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]) 
        fps = round(1.0 / (time.time() - start_time),2)
        #print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        #result = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        #return result, boxes, fps, str(track.track_id), class_name, classes, track_ids
        return result, boxes, fps, classes, track_ids

def deep_parameters_initialization():
    encoder = gdet.create_box_encoder('./model_data/mars-small128.pb', batch_size = 1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.41, None)
    tracker = Tracker(metric)
    return encoder, tracker

def drop_dublicates(path : str):
    """
    Method for dublicates dropping
    input   -> path to images
    """
    print("[INFO] Computing image hashes.")
    try:
        imagePaths = list(paths.list_images(path))
        hashes = {}

        for imagePath in imagePaths:
            image = cv.imread(imagePath)[0:1000,0:1000]
            h = vp.Video_Processing().dhash(image)
            p = hashes.get(h, [])
            p.append(imagePath)
            hashes[h] = p

        for (h, hashedPaths) in hashes.items():
            if len(hashedPaths) > 1:
                if (1):
                    for p in hashedPaths[1:]:
                        os.remove(p)
    except:
        print("[INFO] Some error.")

def video_writer_posuda_vertical(video_path, rotation, settings):
    #>>>>>>>>>>>Work jist unner the docker<<<<<<<<<<<<<<"
    input_video = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc(*settings['video']['codec'])
    width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    total_frames = input_video.get(cv.CAP_PROP_FRAME_COUNT)
    
    image_folder =  settings['links']['screenshots']

    c = 0
    err_count = 0

    with open(settings['NN']['vertical_view_labels_file'], "rt") as f:
        names = f.read().rstrip("\n").split("\n")

    checker_vertical, cameraMatrix_vertical, distCoeffs_vertical = vp.Video_Processing().file_checker()
    encoder_vertical, tracker_vertical = deep_parameters_initialization()
    saved_model_loaded_vertical = tf.saved_model.load("./checkpoints/tiny-416", tags = [tag_constants.SERVING])
    infer_vertical = saved_model_loaded_vertical.signatures['serving_default']

    if input_video.isOpened():

        for i in tqdm(range(int(total_frames - 1))):
            try:
                if (c < total_frames):
                    ret, frame = input_video.read()
                    if (rotation == 1):
                        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
                    
                    '''gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    
                    blurred = cv.medianBlur(gray, 25)
                    minDist = 200
                    param1 = 30 #500
                    param2 = 50 #200 #smaller value-> more false circles
                    minRadius = 45
                    maxRadius = 250 #10

                    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for i in circles[0,:]:
                            cv.circle(frame, (i[0], i[1]), i[2], (255, 215, 0), 6)
                            cv.putText(frame, str(round(i[2] * 0.1059,1)) + str(" cm"), (i[0], i[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 6, cv.LINE_AA, False)

                    try:
                        corners, ids = vp.Video_Processing().marks_detection(checker_vertical, gray, cameraMatrix_vertical, distCoeffs_vertical)
                        #print(corners, ids)

                    except:
                        corners, ids = 0, None
                    result = frame.copy()
                    try:
                        if (corners != 0):
                            result = vp.Drawing().draw_markers_labels(result, corners, ids)
                            
                        else:
                            corners = None
                    except:
                        pass

                    try:
                        result, boxes, fps, classes, track_ids = deep_processing(frame = result, encoder = encoder_vertical, tracker = tracker_vertical, infer = infer_vertical) 

                    except:
                        boxes, fps, classes, track_ids = None, None, None, None 

                    

                    if boxes:
                        centers = []
                        for i in range(len(boxes)):
                            x = int(abs(boxes[i][0] + boxes[i][2]) / 2)
                            y = int(abs(boxes[i][1] + boxes[i][3]) / 2)
                            centers.append([x, y])
                            #result = vp.Drawing().draw_center(result, x, y, radius = 10, color = (0, 0, 255), thickness = 10)
                        
                        if corners is None: #if (corners == 0 or corners == 'None'):
                            print(" ! Chek camera position or marks condition")
                        
                        else:  
                            for i in range(len(track_ids)):
                                if (len(corners) == 4):
                                    if 4 and 2 in ids:
                                        first_index = [j for i in ids.tolist() for j in i].index(2)
                                        second_index = [j for i in ids.tolist() for j in i].index(4)
                                        forth_index = [j for i in ids.tolist() for j in i].index(3)  # WILL CHANGH TO CAMERA COORDINATE!!!!!!!!!!!!!!!!!!!!

                                        p1 = corners[first_index][0][:, 0].mean(), corners[first_index][0][:, 1].mean()
                                        p2 = corners[second_index][0][:, 0].mean(), corners[second_index][0][:, 1].mean()
                                        pc = corners[forth_index][0][:, 0].mean(), corners[forth_index][0][:, 1].mean()

                                        res = vp.Geometry().xy2x_tranformation(p1 = p1, p2 = p2, p3 = centers[i], p4 = pc)
                                        result = vp.Video_Processing().draw_center(image = result, x = int(abs(center[0])), color = (0,0,0),y = int(abs(center[1])), radius = 10)
                                        #print(res)
                                        try:
                                            #self.message(cam_index = "vertical", boxes = boxes, fps = fps, classes = classes, track_id = track_ids)
                                            #self.message(cam_index = "vertical")
                                            #print("vertical", boxes)
                                            pass
                                        except:
                                            print("Some error at: ", datetime.now().strftime("%H:%M:%S"))
                                else:
                                    print(" ! Not enought marks! Chek camera position or marks condition")
                    else:
                        pass     

                    result = vp.Video_Processing().visual_object_orientation(image = result, corners = corners, ids = ids) 

                    
                    '''
                    print("Got frame")
                    cv.imwrite(str(image_folder) + str(c) + str(".JPG"), result)
                    c = c + 1


            except KeyboardInterrupt:
                input_video.release()
                break

            except:

                cv.imwrite(str(image_folder) + str(c) + str(".JPG"), frame)
                c = c + 1

                if (c >= total_frames):
                    input_video.release()
                    print("[INFO] Starting to merge frames")

def send_report_to_email_old(message = None, 
    time_start = None, time_end = None, 
    image = None, port = 465, 
    smtp_server = "smtp.gmail.com", 
    sender_email = "leica.ngc@gmail.com", 
    receiver_email = "ivanzhezhera92@gmail.com",
    password = "solomandra0"):
    """
    Method for email report sending
    """
    context = ssl.create_default_context()
    
    if (time_start != None):
        message = message + "\n" + "\n" + "Process had started at: " + time_start + "\nProcess had finished at: " + time_end
        

    #image = 1
    #ImgFileName = '/data/darknet_AB/chart.png'

    if (image != None):
        image = MIMEImage(img_data, name = os.path.basename(ImgFileName))
        msg.attach(image)
        
    with smtplib.SMTP_SSL(smtp_server, port, context = context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

    print("[INFO] Report had sent to dev email.")


def send_report_to_email(message = None, 
    time_start = None, time_end = None, 
    image = None, port = 465, 
    smtp_server = "smtp.gmail.com", 
    sender_email = "leica.ngc@gmail.com", 
    receiver_email = "ivanzhezhera92@gmail.com",
    password = "solomandra0", 
    subject = "Report",
    filename = None, #"/data/darknet_AB/chart.png"
    body = ""):
    """
    Method for email report sending.
    Report can be sended for 2 adresses.
    In letter can be sended one file, timestamp of start time, end time and text body of letter.
    """
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email

    if (time_start != None):
        body = body + "\n" + "\nProcess had started at: " + time_start + "\nProcess had finished at: " + time_end
    
    message.attach(MIMEText(body, "plain"))
    
    if filename is not None:
        with open(filename, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

            encoders.encode_base64(part)

            part.add_header(
                "Content-Disposition",
                f"attachment; filename = {filename}",
            )

            message.attach(part)

    text = message.as_string()

    context = ssl.create_default_context()
    
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context = context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

    print("[INFO] Report had sent to dev email.")
    

def tiff_to_JPG():  
    base_path = "output/"
    new_path = image_folder
    for infile in os.listdir(base_path):
        print ("file : " + infile)
        read = cv.imread(base_path + infile)
        outfile = infile.split('.')[0] + '.JPG'
        cv.imwrite(new_path + outfile,read,[int(cv.IMWRITE_JPEG_QUALITY), 200])

def zhenya_magic(center_hor_1, center_vert):
        """
        MATH for the fast matching
        """
        hor_1_sector = int(center_hor_1 / 259)
        hor_2_sector = int(center_hor_2 / 259)
        ver_sector = int(center_vert / 194)

        matching_flag = False
        if (hor_1_sector == 1):
            if(ver_sector > 7 and hor_2_sector < 10):
                matching_flag = True
        elif (hor_1_sector == 2):
            if( ver_sector > 5 and hor_2_sector < 9):
                matching_flag = True
        elif (hor_1_sector == 3):
            if (ver_sector > 4 and hor_2_sector < 8):
                matching_flag = True
        elif (hor_1_sector == 4):
            if (ver_sector > 4 and hor_2_sector < 7):
                matching_flag = True
        elif (hor_1_sector == 5):
            if (ver_sector > 2 and hor_2_sector < 8):
                matching_flag = True
        elif (hor_1_sector == 6):
            if (ver_sector > 1 and hor_2_sector < 5):
                matching_flag = True
        elif (hor_1_sector == 7):
            if(ver_sector > 1 and hor_2_sector < 4):
                matching_flag = True

        return matching_flag

def get_coordinates(path_images_vert, path_images_hor, path_file):

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    # FOR VERTICAL
    dekol_class_names = ['zelen_fresh','violleta','klubnika_sad']
    dekol_model_name = './model_v4_08.10.21.[3].h5'#
    dekol_model = load_model(dekol_model_name)

    VERT_LABELS_FILE = "./src/coco_uno.names"
    VERT_CONFIG_FILE = "./src/densenet201_yolo.cfg"
    VERT_WEIGHTS_FILE = "./src/densenet201_yolo_last.weights"
    
    # FOR HORIZONTAL
    HOR_LABELS_FILE = "./src/coco_name.names"
    HOR_CONFIG_FILE = "./src/horizontal_config.cfg"
    HOR_WEIGHTS_FILE = "./src/29.03.21_1.weights"
    
    # NET configuration  - vertical
    vert_net = cv.dnn_DetectionModel(VERT_CONFIG_FILE, VERT_WEIGHTS_FILE)
    vert_net.setInputSize(608, 608)  
    vert_net.setInputScale(1.0 / 255)
    vert_net.setInputSwapRB(True)

    # NET configuration  - angular and horizontal cameras
    hor_net = cv.dnn_DetectionModel(HOR_CONFIG_FILE, HOR_WEIGHTS_FILE)
    hor_net.setInputSize(608, 608)  
    hor_net.setInputScale(1.0 / 255)
    hor_net.setInputSwapRB(True)

    with open(VERT_LABELS_FILE, "rt") as f_h:
        vert_names = f_h.read().rstrip("\n").split("\n")

    with open(HOR_LABELS_FILE, "rt") as f_h:
        hor_names = f_h.read().rstrip("\n").split("\n")

    #vert_images = fnmatch.filter(os.listdir(path_images_vert), '*.JPG')
    hor_images = fnmatch.filter(os.listdir(path_images_hor), '*.JPG')
    
    df_total = pd.DataFrame(columns = ['file_name', 'Class', 'conf', 'x1', 'x2'])

    #vert_images.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    hor_images.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    #print(vert_images)
    print(hor_images)

    for i in tqdm(range(len(hor_images) - 1)): 
        dekol = ""
        try: 
            print(" ")
            '''print(vert_images[i])          
            vert_frame = cv.imread(path_images_vert + vert_images[i])
            vert_classes, vert_confidences, vert_boxes = vert_net.detect(vert_frame, confThreshold = 0.1, nmsThreshold = 0.4)
            
            vert_centers = []
            for vert_classId, vert_confidence, vert_box in zip(vert_classes.flatten(), vert_confidences.flatten(), vert_boxes):
                vert_class_name = vert_names[vert_classId]
                vert_frame = cv.rectangle(vert_frame, vert_box, color = (255, 153, 204), thickness = 25)
                if (vert_box[0] < 1300):
                    vert_centers.append(int(vert_box[1] + vert_box[3]/2))
                for v in range(10):
                    vert_frame = cv.line(vert_frame, (0, 195 * v), (2590, 195 * v), (123,123,123), 3)
                df = pd.DataFrame({'file_name':[vert_images[i]],
                        "Class":[class_name],
                        'conf':[confidence],
                        'x1':[box[1]],
                        'x2':[box[1] + box[3]]
                        })
                df_total = df_total.append(df)


            vert_centers.sort()
            print("Vert centers: ", vert_centers)
            cv.imwrite("/data/yolact/worker_morning_19/img/vertical_labeled/" + str(vert_images[i]), vert_frame)'''

            hor_frame_first = cv.imread(path_images_hor + hor_images[i])
            hor_frame = hor_frame_first
            #hor_frame = hor_frame_sourse
            hor_classes, hor_confidences, hor_boxes = hor_net.detect(hor_frame, confThreshold = 0.1, nmsThreshold = 0.4)
            
            hor_centers = []
            for hor_classId, hor_confidence, hor_box in zip(hor_classes.flatten(), hor_confidences.flatten(), hor_boxes):
                hor_class_name = hor_names[hor_classId]
                #print(vert_class_name, vert_confidence, vert_box)
                hor_frame = cv.rectangle(hor_frame, hor_box, color = (255, 153, 204), thickness = 25)
                hor_frame = cv.putText(hor_frame, hor_class_name, (hor_box[0],hor_box[1] - 50), 6, 2, (255, 153, 204), 3, cv.LINE_AA, False)
                

                cropped =  hor_frame[int(hor_box[1]):int(hor_box[1] + hor_box[3]), int(hor_box[0]):int(hor_box[0] + hor_box[2])]
                img = Image.fromarray(cropped).resize((200,200))
                img = np.expand_dims(img, axis = 0)
                prediction = np.around(dekol_model.predict(img)[0], decimals = 1)
                [float(i) for i in prediction]
                dekol_value = dekol_class_names[prediction.tolist().index(1)]


                hor_frame = cv.putText(hor_frame, dekol_value, (hor_box[0],hor_box[1] - 100), 6, 2, (255, 153, 204), 3, cv.LINE_AA, False)

                hor_centers.append(int(hor_box[0] + hor_box[2]/2))
            
                for h in range(10):
                    hor_frame = cv.line(hor_frame, (259 * h,0), (259 * h,1954), (123,123,123), 3)

            cv.imwrite("/data/yolact/worker_morning_13/img/horizontal_2_labeled/" + str(hor_images[i]), hor_frame)

            hor_centers.sort()
            print("Hor centers: ", hor_centers)

            for q in range(len(vert_centers)):
                for w in range(len(hor_centers)):
                    ver_sector = int(vert_centers[q] / 194)
                    hor_1_sector = int(hor_centers[w] / 259)

                    matching_flag = False
                    if (hor_1_sector == 0):
                        if( ver_sector == 9):
                            matching_flag = True
                    if (hor_1_sector == 2):
                        if( ver_sector > 5 and ver_sector < 9):
                            matching_flag = True
                    elif (hor_1_sector == 3):
                        if (ver_sector > 4 and ver_sector < 8):
                            matching_flag = True
                    elif (hor_1_sector == 4):
                        if (ver_sector  == 4):
                            matching_flag = True
                    elif (hor_1_sector == 5):
                        if (ver_sector > 2 and ver_sector < 5):
                            matching_flag = True
                    elif (hor_1_sector == 6):
                        if (ver_sector > 1 and ver_sector < 5):
                            matching_flag = True
                    elif (hor_1_sector == 7):
                        if(ver_sector == 0):
                            matching_flag = True

                    
                    if(matching_flag == True):

                        cropped =  cv.imread(path_images_hor + hor_images[i])[int(hor_box[1]):int(hor_box[1] + hor_box[3]), int(hor_box[0]):int(hor_box[0] + hor_box[2])]
                        img = Image.fromarray(cropped).resize((200,200))
                        img = np.expand_dims(img, axis = 0)
                        prediction = np.around(dekol_model.predict(img)[0], decimals = 1)
                        [float(i) for i in prediction]
                        dekol_value = dekol_class_names[prediction.tolist().index(1)]
                        print("*********** dekol_value:", dekol_value) 
                        file_name = '/data/yolact/worker_morning_13/cropped/' + str(datetime.now()) + str(" ") + str(dekol_value) + ".JPG"
                        cv.imwrite(file_name, cropped)

                    

            
            #matching = zhenya_magic(center_hor_1, center_hor_2, center_vert)

        
        except Exception as e: print(e)
            

        #print(df_total)

    #df_total.to_csv("/data/yolact/worker_morning_19/img/horizontal_1_labaled/" + "res.csv", header=None, index=None, sep=',')
    
    '''label = confidence
        class_name = self.horizontal_names[classId]
        #label_full = '%s: %s' % (self.horizontal_names[classId], label)

        if (float(label) >= float(settings['NN']['confidence_threshold'])):
            cv.rectangle(result, box, color = (255, 153, 204), thickness = 25)
            #print("Must be rectangle in 2")
            #cv.putText(result, "LBL:" + label_full, (box[0], box[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 4)
                    
            cropped = frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
            h_c, w_c, _ = cropped.shape
            if (w_c < 200):
                w_c = int(img.shape[1] * 2)
                h_c = int(img.shape[0] * 2)
                dim = (w_c, h_c)
                cropped = cv.resize(cropped, dim, interpolation = cv.INTER_AREA)'''

def img_to_video(fps : int, path : str): #".JPG" 
    """
    Method for the image to video convertation
    input    -> fps [video fps]
             -> image_folder
    """
    image_files = []

    filenames = fnmatch.filter(os.listdir(path), '*.JPG')
    if (len(filenames) > 0):
        filenames.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        print("[INFO] Frames appending.", len(filenames))
        
        for i in range (len(filenames)): 
            image_files.append(str(path) + str(filenames[i]))

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps = fps)
        name = str(datetime.now()) + str('.mp4')#'.mp4'
        clip.write_videofile("/data/yolact/1611/results/" + name)

        print("[INFO] Video had created!")

    else:
        print("[INFO] Video had not created! You have not any images.")

def file_correction(path : str):
    """
    Method for the index changing
    """
    path = "/data/darknet_AB/data/RESERV_SP/my/images/"
    _, _, filenames = next(walk(path))
    mask = ".txt"
    masked_list = [filenames for filenames in filenames if mask in filenames]

    
    for i in range(len(masked_list)):
        line_count = 0
        # IF I NEED TO SAVE ALL DATA, JUST CHANGE NUMBERS
        '''df = pd.read_csv(path + masked_list[i], sep=' ' , header = None)
        df[0] = 4
        df.to_csv(path + masked_list[i], header=None, index=None, sep=' ')'''

        with open ( path + masked_list[i], 'r') as file:
            lines = file.readlines()        
            string_list = list(lines[0])
            string_list[0] = "4"
            new_string = "".join(string_list)
            file.close()
            file = open(path + masked_list[i], 'w')
            file.write(new_string)
            file.close()

def camera_calibration():
    """
    Method for the camera calibration
    """
    vp.Video_Processing().camera_calibration()

def real_size():
    checker, cameraMatrix, distCoeffs = vp.Video_Processing().file_checker()
    print(vp.Video_Processing().real_size(checker, None, cameraMatrix, distCoeffs))

def labels_analyzer(path = "/data/darknet_AB/data/total_data/"):
    """
    Make dataframe of all abjects in all images
    """
    df_total = pd.DataFrame(columns = ['Class', 'x', 'y', 'w', 'h'])
    labels = fnmatch.filter(os.listdir(path), '*.txt')
    for i in range(len(labels)):
        try:
            df = pd.read_csv(path + labels[i], sep = ' ', header = None)
            df = df.set_axis(['Class', 'x', 'y', 'w', 'h'], axis = 1)
            df['image'] = labels[i].replace(".txt", ".jpg")
            df_total = df_total.append(df)
        except:
            print(" ! Some error")

    df_total.to_csv("./../" + "list.csv", index = None, sep = ',')
    objects_counters()
    print("===========================")
    print(df_total[df_total['Class'] == 16].head(20))
    #df = pd.read_csv(path + labels[0], sep=' ', header = None)
    #print(df_total['Class'].unique().tolist())

def objects_counters():
    """
    Counter of objects in data base
    """
    df = pd.read_csv("./../" + "list.csv", sep = ',')
    print(len(df))
    print(df.groupby(['Class']).size())
    print(df.head())
    
def find_interesting_classes(path, path_to_save, class_number = 2):
    """
    Clean other label files from other classes
    INPUT    -> path [path to folder with *.JPG &*.txt files], class_number [int value of correct class] 
    OUTPUT   -> --
    """

    labels = fnmatch.filter(os.listdir(path), '*.txt')

    for i in tqdm(range(len(labels))):
        try:
            #name = labels[i].replace(".jpg", ".txt")
            df = pd.read_csv(path + labels[i], sep = ' ', header = None)
            df_human = df.loc[df[0].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])]
            df_human[0].replace({0: class_number, 1: class_number, 2: class_number,
                3: class_number, 4: class_number, 5: class_number, 6: class_number,
                7: class_number, 8: class_number, 9: class_number, 10: class_number,
                11: class_number, 12: class_number, 13: class_number, 14: class_number,
                15: class_number, 16: class_number, 17: class_number, 18: class_number,
                19: class_number, 20: class_number,}, inplace = True)
            #df_ignored = df.loc[df[0].isin([0])]
            #df_human = df_human.append(df_ignored)
            df_human.to_csv(path_to_save + labels[i], index = None, sep = ' ', header = None)
        except:
            pass

def merger_of_csv(path = "./../"):
    """
    Собираем все списки в один
    """
    df_train = pd.read_csv(path + "train_annotations.csv", sep = ',', header = None)
    df_test = pd.read_csv(path + "test_annotations.csv", sep = ',', header = None)
    df_valid = pd.read_csv(path + "val_annotations.csv", sep = ',', header = None)
    df_total = pd.DataFrame()
    df_total = df_total.append(df_train)
    df_total = df_total.append(df_test)
    df_total = df_total.append(df_valid)
    df_total.to_csv("./../" + "china.csv", index = None, sep = ',', header = None)
    #print(df_total.head())
    print("[INFO] Total list is ready!")

def video_recovery_function(path = '/data/RECOVERY/NA_OTPRAVKU/', image_path = '/data/yolact/Pictures/'):
    
    videos = fnmatch.filter(os.listdir(path), '*.avi')
    videos.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    c = 0
    
    for i in range (len(videos)):
        input_video = cv.VideoCapture(path + videos[i])
        total_frames = input_video.get(cv.CAP_PROP_FRAME_COUNT)
        if input_video.isOpened():
            for i in tqdm(range(int(total_frames - 1))):

                try:
                    if (c < total_frames):
                        ret, frame = input_video.read()
                        cv.imwrite(str(image_path) + str(c) + str(".JPG"), frame)
                        c = c + 1
    
                except KeyboardInterrupt:
                    input_video.release()
                    break
    
                except:
                    c = c + 1
                    if (c >= total_frames):
                        input_video.release()
                        

    print("Done!")

def csv2txt_transformation(path = './../'):
    """
    В списке всех записей выбираем только людей и разбиваем на лейблы
    """
    df = pd.read_csv(path + "china.csv", sep = ',', header = None)
    df[0] = df[0].str.split('/').str.get(2)
    df = df[df[5] == 'Pedestrian']
    print(df.head())
    print()
    print(df[5].unique().tolist())
    print()
    print(len(df))
    print()
    print(df.groupby([5]).size())

    for i in tqdm(range(len(df[0].unique().tolist()))):
        df_small = df[df[0] == df[0].unique().tolist()[i]]
        name = df[0].unique().tolist()[i].replace(".jpg", "")

        img = cv.imread("./china/" + str(df[0].unique().tolist()[i]))
        height, width, channel = img.shape

        del df_small[0]
        column_names = [5, 1, 2, 3, 4]
        df_small[5] = "0"
        
        df_small[1] = round(df_small[1]/width, 6)   
        df_small[2] = round(df_small[2]/height, 6)   
        x_delta = df_small[3]/width - df_small[1]/width
        y_delta = df_small[4]/height - df_small[2]/height
        df_small[3] = round(df_small[3]/width - df_small[1],6) 
        df_small[4] = round(df_small[4]/height - df_small[2],6)
        df_small = df_small.reindex(columns = column_names)
        df_small.to_csv("./labels/" + str(name) +".txt", index = None, sep = ' ', header = None)

def counter(path = './../darknet_AB/data/total_data/'):
    labels = fnmatch.filter(os.listdir(path), '*.txt')
    images = fnmatch.filter(os.listdir(path), '*.jpg')
    images_b = fnmatch.filter(os.listdir(path), '*.JPG')

    print("labels: ",len(labels),"jpg: ", len(images), "JPG: ", len(images_b))

def file_correction_2(path = '/data/yolact/DEKOL/classification/buf/', value = 2):
    # changing class numbers by constant value
    labels = fnmatch.filter(os.listdir(path), '*.txt')
    for i in tqdm(range(len(labels))):
        try:
            df = pd.read_csv(path + labels[i], sep = ' ', header = None)
            df[0] = value
            #print(df)
            df.to_csv(path + labels[i], index = None, sep = ' ', header = None)
            #print(labels[201])
        except:
            pass
    print(' * Classes have changed!')

def file_correction_3(path = '/data/yolact/DEKOL/classification/buf/'):
    # changing class numbers by matrix
    labels = fnmatch.filter(os.listdir(path), '*.txt')
    for i in tqdm(range(len(labels))):
        try:
            df = pd.read_csv(path + labels[i], sep = ' ', header = None)
            df[0].replace({15: 3, 16: 1, 17: 2,}, inplace = True)
            
            #print(df)
            df.to_csv(path + labels[i], index = None, sep = ' ', header = None)
            #print(labels[201])
        except:
            pass
    print(' * Classes have changed!')

def comma2space(path = './china/'):
    labels = fnmatch.filter(os.listdir(path), '*.txt')
    for i in tqdm(range(len(labels))):
        fin = open(path + str(labels[i]), "rt")
        fout = open("./output/" + str(labels[i]), "wt")
        for line in fin:
            fout.write(line.replace(',', ' '))
        fin.close()
        fout.close()

def blured_images_detection(path = '/data/darknet_AB/data/total_data_SP/'):
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    counter = 0
    fm_list = []
    for i in tqdm(range(len(images))):
        image = cv.imread(str(path) + str(images[i]))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        fm = round(cv.Laplacian(gray, cv.CV_64F).var(),2)
        if (fm < 100):
            #print(images[i], fm)
            counter = counter + 1
        fm_list.append(fm)
    print("Total: ", i)
    print("Blured:", counter)
    part = round(counter/i*100, 2)
    print("Part:  ", part, "%")
    return fm_list

def small_objects_images_detection(path):
    labels = fnmatch.filter(os.listdir(path), '*.txt')
    areas = []
    for i in tqdm(range(len(labels))): #
        try:
            df = pd.read_csv(path + labels[i], sep = ' ', header = None)
            df[5] = df[4] * df[3]
            areas.extend(df[5].tolist())

        except:
            print("[INFO] No data")
    return areas

def data_hist(path = '', path_to_save = ''):
    import scipy.stats as st
    x = small_objects_images_detection(path = path) #blured_images_detection()
    plt.hist(x, bins = 1000, label = "Area ")
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    #kde_xs = np.linspace(mn, mx, 5)
    plt.axvline(x = .0001, color = 'r' )
    #kde = st.gaussian_kde(x)
    plt.xlim([mn, 0.1])
    #plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    plt.legend(loc = "upper left")
    plt.ylabel('Quantity')
    plt.xlabel('Area')
    plt.title("Histogram");
    plt.savefig(path_to_save, bbox_inches = 'tight')
    
def files_cleaning(path = None, size_level = 0.001, blur_level = 110):
    """ 
    Data cleaning method. 
    """
    if (path == None):
        print("[INFO] Please, set correct path")

    else:
        print("[INFO] Cleaning images with small objects")
        labels = fnmatch.filter(os.listdir(path), '*.txt')
        
        print("[INFO] Total quantity: ", len(labels), "(*.txt)")
        counter = 0
        for i in tqdm(range(len(labels))): #
            try:
                df = pd.read_csv(path + labels[i], sep = ' ', header = None)
                df[5] = df[4] * df[3]
                if (size_level > df[5].min()):
                    counter = counter + 1
                    os.remove(path + labels[i])
                    os.remove(path + str(labels[i]).replace(".txt", ".JPG")) 
                    print("[INFO] " + str(labels[i]) + " and it image have deleted" )
            except:
                pass

        print("[INFO] Had deleted: ", counter , " pairs.")
        ''''print("[INFO] Cleaning blured images")
        images = fnmatch.filter(os.listdir(path), '*.JPG')
        for i in tqdm(range(len(images))):
            image = cv.imread(str(path) + str(images[i]))
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            fm = round(cv.Laplacian(gray, cv.CV_64F).var(),2)
            if (fm < blur_level):
                os.remove(path + images[i])
                os.remove(path + str(images[i]).replace(".JPG", ".txt"))
        '''
        labels = fnmatch.filter(os.listdir(path), '*.txt')
        print("[INFO] Final quantity: ",len(labels), "(*.txt)")

def file_format_regester_checker(path = ''):
    """
    Method for file format cheacking. Needed for data preproccing for model training
    Will chaing jpeg and jpg to JPG format
    INPUT   -> path of images
    OUTPUT  -> --
    """
    file_name_list = os.listdir(path)
    file_formats_list = [x.split(".")[1] for x in file_name_list]
    file_formats_list_without_dublicates = list(dict.fromkeys(file_formats_list))
    print("[INFO] In folder exists files with next file formats: ", file_formats_list_without_dublicates)
    
    if ("jpg" in file_formats_list_without_dublicates or "jpeg" in file_formats_list_without_dublicates):
        answer = input("Detected uncorrect format. Do You want to make change of all images to *.JPG format? \nPress [Y/N] to continue:  ")
        if (answer == "y" or answer == "Y" or answer == "YES" or answer == "yes" or answer == "Yes"):
            #list_with_correct_format = [string.replace(".jpeg",".JPG").replace(".jpg",".JPG") for string in file_name_list]
            os.chdir(path)
            for count, f in enumerate(file_name_list):
                f_name, f_ext = os.path.splitext(f)
                if (f_ext == '.jpeg' or f_ext == '.jpg'):
                    f_ext = '.JPG'
                    new_name = f'{f_name}{f_ext}'
                    os.rename(f, str(new_name))

            print("[INFO] Register of images format was changed to *.JPG")

        elif(answer == "n" or answer == "N" or answer == "NO" or answer == "no" or answer == "No"):
            print("[INFO] All formats will be without changing")

        else:
            print("[INFO] Uncorrect input value. All formats will be without changing")
    else:
        print("[INFO] All files with correct format")

def combine_two_color_images_with_anchor(image1, image2, anchor_y, anchor_x):
    foreground, background = image1.copy(), image2.copy()
    # Check if the foreground is inbound with the new coordinates and raise an error if out of bounds
    background_height = background.shape[0]
    background_width = background.shape[1]
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    if foreground_height+anchor_y > background_height or foreground_width + anchor_x > background_width:
        raise ValueError(" ! The foreground image exceeds the background boundaries at this location")
    
    alpha = 0.99

    # do composite at specified location
    start_y = anchor_y
    start_x = anchor_x
    end_y = anchor_y + foreground_height
    end_x = anchor_x + foreground_width
    blended_portion = cv.addWeighted(foreground,
                alpha,
                background[start_y:end_y, start_x:end_x,:],
                1 - alpha,
                0,
                background)
    background[start_y:end_y, start_x:end_x,:] = blended_portion
    return background

def blurring_dekol(path, x1, y1, x2, y2):
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    for i in range(len(images)):
        img = cv.imread(path + images[i])
        blurred_img = cv.GaussianBlur(img, (21, 21), 0)
        mask = np.zeros((512, 512, 3), dtype = np.uint8)
        #mask = cv.circle(mask, (258, 258), 100, np.array([255, 255, 255]), -1)
        mask = cv.rectangle(mask, (x1,y1), (x2,y2), np.array([255, 255, 255]), -1)
        out = np.where(mask!=np.array([255, 255, 255]), img, blurred_img)
        cv.imwrite(path + images[i], out)

def add_extra_pairs(path):
    print("[INFO] Clone generation")
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    labels = fnmatch.filter(os.listdir(path), '*.txt')

    for i in tqdm(range(len(images))):
        try:
            label = pd.DataFrame()
            image = vp.Image_augmentation().image_horizontal_flip(image = cv.imread(str(path) + str(images[i])))
            df = pd.read_csv(str(path) + str(images[i])[:-3] + str("txt"), sep = " ", header = None)
            df[1] = 1 - df[1]
            df[2] = 1 - df[2]
            time = str(datetime.now())
            image_name = str(path) + time + str(".JPG")
            cv.imwrite(image_name, image)
            df.to_csv(str(path) + time + '.txt', sep = ' ', index = False, header = False)
        except:
            print(" ! Some error with clone generation")

def add_extra_images(path, quantity):
    """
    Method for manual adding extra images for dataset. Add flipped and darkness images
    """
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    start_quantity = len(images)
    if ((2 * start_quantity) < quantity):
        delta = start_quantity

    else:
        delta = quantity - start_quantity

    images_for_flip = random.sample(images, delta) 
    #print("images_for_flip", images_for_flip)
    print(len(images_for_flip))

    for i in range(len(images_for_flip)):
        try:
            image = cv.imread(str(path) + images_for_flip[i])
            # Flipping
            #image = vp.Image_augmentation().image_horizontal_flip(image = image)
            # Make darkness
            image = vp.Image_augmentation().image_gamma_correction(image = image, gamma = random.uniform(0.3, 0.8))
            # Decrease quality of images
            image = vp.Image_augmentation().image_file_resizing(image = image)

            cv.imwrite(str(path) + str(datetime.now()) + str(".JPG"), image)
            print("[INFO] Darkny image")
        except:
            print("[INFO] Some error")

def dekol_files_auto_preparation(images_path = '', train_images_path = '', val_images_path = '', 
    eval_images_path = '', access_rights = 0o755, iterations : int = 200, file_format = '*.png'):
    ''' Method for files preparation to be possible dekol model training
    input  -> path with '*.png' images
           -> path with train, valid and evaluate folders
    output -> 0 if all is done
           -> 1 if something wrong'''

    #try:
    if(True):
        
        dekol_list = fnmatch.filter(os.listdir(images_path), file_format)
        quantity = len(dekol_list)
        print("[INFO] Images generation is starting...")
        for path in [train_images_path, val_images_path, eval_images_path]:
            _, folders, filenames = next(walk(path))

            if (len(filenames) > 0 or len(folders) > 0):
                print(" ! Folders need to be cleaned")
                recursive_total_cleaning(path = path)
                print("[INFO] Folders have cleaned")

            print("    - preparation: ", path)
            for dir_name in range(quantity):
                print(" >>> ", dir_name)
                os.mkdir(str(path) + str(dir_name) + str('/'), access_rights)

                image_transformation(path = images_path + str(dekol_list[dir_name]), class_name = str(dir_name) + str("_"), 
                    iterations = int(iterations), output_path = str(path) + str(dir_name))

            iterations = int(int(iterations) / 2)

        os.system("chmod -R 777 " + "/data/yolact/DEKOL/classification/")
        print("[INFO] Dekol files auto preparation process have done!")
        
def image_transformation_yolo_light(path = '/data/yolact/GEN_DATA/current/'):
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    for i in tqdm(range(len(images))):
        image = cv.imread(path + images[i])
        image = vp.Image_augmentation().image_on_image(image = image)

        data = "0,0.5233,0.6061,0.5616,0.4322"

        f = open(str('/data/yolact/GEN_DATA/result/') + str(images[i])[:-4] + str(".txt"), "a")
        f.write(data)
        f.close()
        cv.imwrite(str('/data/yolact/GEN_DATA/result/') + str(images[i])[:-4] + str('.JPG'), image)

    print("[INFO] Done!")
    
def image_transformation_yolo(iterations = 20, image_path = "/data/yolact/DEKOL/all_dekol/3.png", 
                                path = "/data/yolact/DEKOL/classification/white_models/", 
                                out_path = "/data/yolact/DEKOL/classification/imds_small/", class_name = "4"
                                ):
    # images with white objects
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    for i in tqdm(range(len(images))):
        try:
            # Reading BACKGRAUND image
            image = cv.imread(path + str(images[i]))
            oH,oW = image.shape[:2]
            image = np.dstack([image, np.ones((oH,oW), dtype = "uint8") * 255])

            # Reading LOGO image
            lgo_img = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)

            scl = 50
            w = int(lgo_img.shape[1] * scl / 100)
            h = int(lgo_img.shape[0] * scl / 100)
            dim = (w,h)
            lgo = cv.resize(lgo_img, dim, interpolation = cv.INTER_AREA)
            lH,lW = lgo.shape[:2]

            if (oW <= lW):
                k = (1 - lW / oW) * 2
                oW = oW * k
                oH = oH * k
                image = cv.resize(image, (oW, oH), interpolation = cv.INTER_AREA)

            ovr = np.zeros((oH,oW,4), dtype = "uint8")
            ovr[int(oH * 0.45):int(oH * 0.45 + h), int(oW * 0.25):int(oW * 0.25 + w)] = lgo
            final = image.copy()
            #final = vp.Image_augmentation().image_channel_changing(final)            
            image = cv.addWeighted(ovr,0.2,final,0.8,0)
            for j in tqdm(range(iterations)):
                try:
                    center_x = ((oW * 0.25) + (oW * 0.25 + w)) / 2
                    center_y = ((oH * 0.45) + (oH * 0.45 + h)) / 2
                    object_w = w
                    object_h = h

                    flip_flag = bool(random.getrandbits(1))
                    zoom_value = random.uniform(1, 1.01)
                    angle = random.uniform(-2, 2)
                    horizontal_ratio = random.uniform(0.01, 0.2)
                    vertical_ratio = random.uniform(0.01, 0.2)

                    image_iter = vp.Image_augmentation().image_horizontal_flip(image = image, flag = flip_flag)
                    #image_iter = vp.Image_augmentation().image_zoom(image = image_iter, zoom_value = zoom_value)
                    #image_iter = vp.Image_augmentation().image_resizing(image = image_iter)
                    image_iter = vp.Image_augmentation().image_rotation(image = image_iter, angle = angle)
                    image_iter = vp.Image_augmentation().image_noise(image = image_iter)
                    image_iter = vp.Image_augmentation().image_gamma_correction(image = image_iter, gamma = random.uniform(0.2, 1.5))
                    image_iter = vp.Image_augmentation().image_blur(image = image_iter, k_param = random.randint(1, 3))
                    image_iter, horizontal_ratio = vp.Image_augmentation().image_horizontal_shift(image = image_iter, ratio = horizontal_ratio) 
                    image_iter, vertical_ratio = vp.Image_augmentation().image_vertical_shift(image = image_iter, ratio = vertical_ratio)
                    image_iter = vp.Image_augmentation().channel_shuffle(image = image_iter, param_1 = random.uniform(0.1, 0.35))
                    image_iter = vp.Image_augmentation().add_value(image = image_iter)
                    #image_iter = vp.Image_augmentation().multiply_hue(image = image_iter)
                    #image_iter = vp.Image_augmentation().change_color_temperature(image = image_iter)
                    # image_iter = vp.Image_augmentation().simple_image_resizing(image = image_iter)

                    #image_iter = vp.Image_augmentation().sigmoid_contrast(image = image_iter)
                    
                    df = pd.DataFrame(columns = ['class_name', 'center_x', 'center_y', 'object_w', 'object_h'])
                    

                    if (flip_flag == True):
                        center_x = 1 - center_x / oW
                        center_y = 1 - center_y / oH

                    elif(flip_flag == False):
                        center_x = center_x / oW
                        center_y = center_y / oH
                    
                    center_x = center_x * (horizontal_ratio + 1)
                    center_y = center_y * (vertical_ratio + 1)

                    #image_iter = vp.Drawing().draw_center(image = image_iter, x = int(oW * center_x), y = int(oH * center_y), radius = int(object_w / 2 * (abs(horizontal_ratio) + 1)))
                    #image_iter = vp.Drawing().draw_center(image = image_iter, x = int(oW * center_x), y = int(oH * center_y), radius = int(object_h / 2 * (abs(vertical_ratio) + 1)))
                    #image_iter = vp.Drawing().draw_center(image = image_iter, x = int(oW * center_x), y = int(oH * center_y), radius = 5)


                    object_w = int(object_w * (abs(horizontal_ratio) + 1)) / oW
                    
                    if (object_w > 1):
                        object_w = 1

                    object_h = int(object_h * (abs(vertical_ratio) + 1)) / oH

                    if (object_h > 1):
                        object_h = 1
                    
                    uniq_file_time = str(datetime.now())
                    cv.imwrite(str(out_path) + uniq_file_time + ('.JPG'), image_iter)
                    
                    df.loc[0] = [class_name, center_x, center_y, object_w, object_h]
                    df.to_csv(str(out_path) + uniq_file_time + '.txt', sep = ' ', index = False, header = False)


                except:
                    print(" ! Some error inside iterrations loop")
            

        except:
            print(" ! Some error inside dekols loop")

def image_transformation(path, class_name, iterations, output_path):
    """Metthod for image generation (augmentation) for the classification task"""    
    try:
        for j in tqdm(range(iterations)):  
            image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            #===================================================
            #h, w, _ = image.shape
            #y = random.randint(0, w - h)
            #image = image[y:y + h, 0:h]
            
            #===================================================

            h, w, channels = image.shape
            backgraund = np.ones((h, w, 3))              
            
            try:
                # cut backgraund
                trans_mask = image[:, :, 3] == 0
                # backgraund generation
                mask_color = [random.randint(200, 255), random.randint(200, 255), random.randint(200, 255), 255]

                # for spetial backgraund generation
                #mask_color = [cv.cvtColor(random.randint(0, 359), random.randint(0, 6), 100,cv2.COLOR_HSV2RGB), 255]
                
                image[trans_mask] = mask_color


            except:
                pass
            
            # resize image and changing
            image_buf = image.copy()
            image_buf = vp.Image_augmentation().image_zoom(image = image_buf, zoom_value = random.uniform(1, 1.01) )
            #image_buf = vp.Image_augmentation().image_resizing(image = image_buf, scale_percent = random.randint(60, 100))

            #image_buf = vp.Image_augmentation().image_rotation(image = image_buf, angle = random.uniform(-15, 15))
            #image_buf = vp.Image_augmentation().image_noise(image = image_buf)
            #image_buf = vp.Image_augmentation().image_gamma_correction(image = image_buf, gamma = random.randint(1, 5))
            #image_buf = vp.Image_augmentation().image_blur(image = image_buf, k_param = random.randint(1, 5))


            #=================
            flip_flag = bool(random.getrandbits(1))
            zoom_value = random.uniform(0.85, 1)
            angle = random.uniform(-15, 15)
            horizontal_ratio = random.uniform(0.01, 0.2)
            vertical_ratio = random.uniform(0.01, 0.2)

            image_buf = vp.Image_augmentation().image_horizontal_flip(image = image_buf, flag = flip_flag)
            #image_buf = vp.Image_augmentation().image_zoom(image = image_buf, zoom_value = zoom_value)
            image_buf = vp.Image_augmentation().image_resizing(image = image_buf)
            image_buf = vp.Image_augmentation().image_rotation(image = image_buf, angle = angle)
            image_buf = vp.Image_augmentation().image_noise(image = image_buf)
            image_buf = vp.Image_augmentation().image_gamma_correction(image = image_buf, gamma = random.uniform(0.7, 1.3))
            image_buf = vp.Image_augmentation().image_blur(image = image_buf, k_param = random.randint(1, 3))
            image_buf, horizontal_ratio = vp.Image_augmentation().image_horizontal_shift(image = image_buf, ratio = horizontal_ratio) 
            image_buf, vertical_ratio = vp.Image_augmentation().image_vertical_shift(image = image_buf, ratio = vertical_ratio)
            ##image_buf = vp.Image_augmentation().channel_shuffle(image = image_buf, param_1 = random.uniform(0.1, 0.35))
            image_buf = vp.Image_augmentation().add_value(image = image_buf)
            ##image_buf = vp.Image_augmentation().multiply_hue(image = image_buf)
            ##image_buf = vp.Image_augmentation().change_color_temperature(image = image_buf)
            # image_buf = vp.Image_augmentation().simple_image_resizing(image = image_buf)
            image_buf = vp.Image_augmentation().sigmoid_contrast(image = image_buf)
            #=================
            #image_buf, _ = vp.Image_augmentation().image_horizontal_shift(image = image_buf, ratio = random.uniform(0.01, 0.8)) 
            #image_buf, _ = vp.Image_augmentation().image_vertical_shift(image = image_buf, ratio = random.uniform(0.01, 0.8))
            ##image_buf = cv.cvtColor(image_buf, cv.COLOR_BGR2GRAY)

            h, w, ch  = image_buf.shape

            if (random.choice([True, False])):
                if (w > h):
                    x = random.randint(0, w - h)
                    crop_img = image_buf[0:h, x:x + h]
            else:
                crop_img = image_buf

            '''

            depending = h / w
         
            pts1 = np.float32([[0,  0],[w, 0],[0, h],[w, h]])

            ############ 
            #          # 
            # p1 -> p2 #
            # p3 -> p4 #
            #          #
            ############

            random_level = 0.001

            p1 = [0 + w * random.uniform(0, random_level), 0 + h * random.uniform(0, random_level)]
            p2 = [w + w * random.uniform(0, random_level), 0 + h * random.uniform(0, random_level)]
            p3 = [0 + w * random.uniform(0, random_level), h + h * random.uniform(0, random_level)]
            p4 = [w + w * random.uniform(0, random_level), h + h * random.uniform(0, random_level)]

            pts2 = np.float32([p1, p2, p3, p4])

            #print(pts2)
            if (w > h):
                new_w = int(np.max(pts2))
                new_h = int(new_w * depending) 

            elif(h >= w):
                new_h = int(np.max(pts2))
                new_w = int(new_h / depending) 
                    
            # If we are creating DB for the empty image
            try:
                image_new = vp.Image_augmentation().image_trasformation(image_buf, pts1, pts2, new_w, new_h)
                    
            except:
                image_new = image_buf

            center_x = round((p1[0] + p2[0] + p3[0] + p4[0]) / 4 / w, 6) 
            center_y = round((p1[1] + p2[1] + p3[1] + p4[1]) / 4 / h, 6)
        
            w_part = round((max(p1[0], p2[0], p2[0], p3[0]) - min(p1[0], p2[0], p2[0], p3[0])) / w, 6)
            h_part = round((max(p1[1], p2[1], p2[1], p3[1]) - min(p1[1], p2[1], p2[1], p3[1])) / h, 6)

            if (w_part > 1):
                w_part = 1
            if (h_part > 1):
                h_part = 1    

            image_new = cv.cvtColor(image_new, cv.COLOR_BGRA2BGR)
            #image_new = cv.cvtColor(image_new, cv.COLOR_BGR2GRAY)
            if(random.randint(0, 1)):
                image_new = image_new[int(0.2*h):int(1.2*h), int(0.2*w):int(1*w)]
                #print(image_new.shape)
                        
                min_random_edje = 0
                max_random_edje = int(image_new.shape[1] - image_new.shape[0]) 
                w_start = random.randint(min_random_edje, max_random_edje)
                w_end = w_start + image_new.shape[0]

                #print(image_new.shape, w_start,w_end, min_random_edje, max_random_edje)
                crop_img = image_new[0:image_new.shape[0], w_start:w_end]
            else:
                crop_img = image_new'''

            width = 256
            height = 256

            dsize = (width, height)
            crop_img = cv.resize(crop_img, dsize)
            #crop_img = image_buf

            

            file_creation_time = str(datetime.now())
                        
            cv.imwrite(str(output_path)  + str("/") + file_creation_time + str('.JPG'), crop_img)
    except:
        print("! Had not prepared")
            
def image_cutter_4_classification_by_model_prediction(
    path_input = '/data/yolact/1611/1_camera/', 
    path_output = '/data/yolact/1611/cropped/', 
    format_file = '*.JPG'):
    
    print(" Model init")

    HORIZONTAL_LABELS_FILE = "./src/classes.names"
    HORIZONTAL_CONFIG_FILE = "./src/config_.cfg"
    HORIZONTAL_WEIGHTS_FILE = "./src/weights.weights"

    with open(HORIZONTAL_LABELS_FILE, "rt") as f_h:
        horizontal_names = f_h.read().rstrip("\n").split("\n")

    horizontal_net = cv.dnn_DetectionModel(HORIZONTAL_CONFIG_FILE, HORIZONTAL_WEIGHTS_FILE)
    horizontal_net.setInputSize(608, 608)  # <- FOR YOLOV4
    horizontal_net.setInputScale(1.0 / 255)
    horizontal_net.setInputSwapRB(True)
    
    print("[INFO] Model is ready!")
    
    print("[INFO] Images preparation:")
    images = fnmatch.filter(os.listdir(path_input), format_file)
    for i in tqdm(range(len(images))):
        image = cv.imread(path_input + images[i])
        try:
            classes, confidences, boxes = horizontal_net.detect(image, confThreshold = 0.1, nmsThreshold = 0.4)
            if len(classes) > 0:
                for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                    label = confidence
                    if (float(label) >= 0.1):
                        cropped = image[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
                        name = str(path_output) + str(datetime.now()) + str(".JPG")
                        h,w,_ = cropped.shape
                        if (w > 0.8 * h):
                            cv.imwrite(name, cropped)


        except Exception as e: 
            print(e)
            
def image_cutter_4_classification_by_size_frame(
    path_input = '/data/yolact/DEKOL/classification/imds_nano_3/train/2/', 
    path_output = '/data/yolact/labels/', 
    format_file = '*.JPG'):
    """
    Method for writing parameters by size
    INPUT   -> path_input 
            -> path_output
            -> format_file 
    OUTPUT  -> --
    """
    print("[INFO] Images preparation")
    
    images = fnmatch.filter(os.listdir(path_input), format_file)
    for i in tqdm(range(len(images))):
        image = cv.imread(path_input + images[i])
        h,w,_ = image.shape
        try:
            cropped = image[int(h * 0.3):int(h * 0.99), int(w * 0.01):int(w * 0.99)]
            name = str(path_output) + str(datetime.now()) + str(".JPG")
            cv.imwrite(name, cropped)


        except Exception as e: 
            print(e)
            print("[INFO] Error with file: " + str(path_input + images[i]) + " preparation!")

    print("[INFO] Preparation has Done!")

def add_filter(path = "/data/yolact/input/"): #"/data/yolact/DEKOL/classification/imds_nano_3/train/2/"
    """ 
    Method for automatic adding filter over the image
    """
    images = fnmatch.filter(os.listdir(path), '*.JPG')#'*.JPG'
    for i in tqdm(range(len(images))):
        img = cv.imread(path + images[i],1)
        img = vp.Image_augmentation().image_gamma_correction(img, gamma = random.uniform(1.0, 2.5))
        img = vp.Image_augmentation().set_saturation(image = img, persent = random.uniform(0.3, 0.9))
        img = vp.Image_augmentation().set_value(image = img, persent = random.uniform(0.9, 1.9))
        img = vp.Image_augmentation().image_noise(image = img, max_noise_level = random.uniform(0.1, 0.7))
        img = vp.Image_augmentation().image_blur(image = img, k_param = random.randint(1,9))
        #img = vp.Image_augmentation().image_zoom(image = img, zoom_value = random.uniform(1.0, 1.5))
        #img = vp.Image_augmentation().image_rotation(image = img, angle = 20)
        cv.imwrite(path + str("1_") + images[i], img)
    print("[INFO] Image processing has done!")

def image_cutter_4_classification_labels (path_input = '', path_output = ''):
    """
    Method for images cuuting by label files. 
    Needed for classification model training data preparation.
    """
    try:
        images = fnmatch.filter(os.listdir(path_input), '*.JPG')
        labels = fnmatch.filter(os.listdir(path_input), '*.txt')

        for i in tqdm(range(len(images))):
            x_center = 0; y_center = 0; w = 0; h = 0    
        
            image = cv.imread(path_input + images[i])
            h,w,_ = image.shape
            try:
                with open(path_input + images[i].split(".")[0] + '.txt', 'r') as file:
                    data = file.read().replace('\n', '')
                    x_center_relative = float(data.split(" ")[1])
                    y_center_relative = float(data.split(" ")[2])
                    w_relative = float(data.split(" ")[3])
                    h_relative = float(data.split(" ")[4])


                if (x_center_relative > 0 and y_center_relative > 0):
                    image = image[int((y_center_relative - h_relative / 2) * h):int((y_center_relative + h_relative / 2) * h), int((x_center_relative - w_relative / 2) * w):int((x_center_relative + w_relative / 2) * w)]
                    h, w, _ = image.shape
                    #print(path + images[i], w, h)
                    #print("/data/yolact/labels/" + str(i) + str(".JPG"), h, w)
                    cv.imwrite(path_output + str(images[i]) + str(".JPG"), image)
            except:
                pass

    except:
        print(" ! Error")

def image_transformation_old(path = './DEKOL/png/', class_name ="img_", iterations = 5000, output_path = './DEKOL/classification/imds_nano_9/'):
    
    images = fnmatch.filter(os.listdir(path), '*.png')
    # images = fnmatch.filter(os.listdir(path), '*.JPG')
    counter = 0
    if (len(images) > 0):
        
        h, w, channels = cv.imread(path + images[0]).shape
        backgraund = np.ones((h, w, 3))
        
        for j in tqdm(range(iterations)):
            for i in range(len(images)):
                #print("read", counter)
                
                image = cv.imread(str(path) + str(images[i]), cv.IMREAD_UNCHANGED)
                class_img = str(images[i])[:-4] + '/'
                

                try:
                    # cut backgraund
                    trans_mask = image[:, :, 3] == 0
                    # backgraund generation
                    mask_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255]
                    image[trans_mask] = mask_color
                
                except:
                    pass

                # resize image and changing
                image_buf = image.copy()
                image_buf = vp.Image_augmentation().image_zoom(image = image_buf, zoom_value = random.uniform(0.9, 1))
                image_buf = vp.Image_augmentation().image_resizing(image = image_buf, scale_percent = random.randint(70, 99))

                image_buf = vp.Image_augmentation().image_rotation(image = image_buf, angle = random.uniform(-3, 3))
                # image_buf = vp.Image_augmentation().image_noise(image = image_buf)
                image_buf = vp.Image_augmentation().image_gamma_correction(image = image_buf, gamma = random.randint(1, 2))
                # image_buf = vp.Image_augmentation().image_blur(image = image_buf, k_param = random.randint(1, 4))

                # image_buf = vp.Image_augmentation().channel_shuffle(image = image_buf, param_1 = random.uniform(0.1, 0.35))
                # image_buf = vp.Image_augmentation().add_value(image = image_buf)
                # image_buf = vp.Image_augmentation().multiply_hue(image = image_buf)
                # image_buf = vp.Image_augmentation().change_color_temperature(image = image_buf)
                # image_buf = vp.Image_augmentation().simple_image_resizing(image = image_buf)
                # image_buf = vp.Image_augmentation().sigmoid_contrast(image = image_buf)
                image_buf = vp.Image_augmentation().image_change_contrast(image = image_buf)

                h, w, ch  = image_buf.shape

                depending = h / w
     
                pts1 = np.float32([[0,  0],[w, 0],[0, h],[w, h]])

                ############ 
                #          # 
                # p1 -> p2 #
                # p3 -> p4 #
                #          #
                ############
                
                random_level = 0.05

                p1 = [0 + w * random.uniform(0, random_level), 0 + h * random.uniform(0, random_level)]
                p2 = [w + w * random.uniform(0, random_level), 0 + h * random.uniform(0, random_level)]
                p3 = [0 + w * random.uniform(0, random_level), h + h * random.uniform(0, random_level)]
                p4 = [w + w * random.uniform(0, random_level), h + h * random.uniform(0, random_level)]

                pts2 = np.float32([p1, p2, p3, p4])

                #print(pts2)
                if (w > h):
                    new_w = int(np.max(pts2))
                    new_h = int(new_w * depending) 

                elif(h >= w):
                    new_h = int(np.max(pts2))
                    new_w = int(new_h / depending) 
                
                # If we are creating DB for the empty image

                try:
                    image_new = vp.Image_augmentation().image_trasformation(image_buf, pts1, pts2, new_w, new_h)
                
                except:
                    image_new = image_buf
                    
                ''' 
                image_buf_A = cv.cvtColor(image_new, cv.COLOR_BGR2BGRA)   

                #print(image.shape)
                #print(image_new.shape)
                HB, WB, _ = image.shape
                trans_mask = image[:, :, 3] > 0 
                image[trans_mask] = mask_color
                #h, w, _ = image.shape
                anchor_x = int(random.uniform(50, WB * .25))
                anchor_y = int(random.uniform(70, HB * .25))
                image_new = combine_two_color_images_with_anchor(image_buf_A, image, anchor_y, anchor_x)
                '''
                center_x = round((p1[0] + p2[0] + p3[0] + p4[0]) / 4 / w, 6) 
                center_y = round((p1[1] + p2[1] + p3[1] + p4[1]) / 4 / h, 6)
    
                w_part = round((max(p1[0], p2[0], p2[0], p3[0]) - min(p1[0], p2[0], p2[0], p3[0])) / w, 6)
                h_part = round((max(p1[1], p2[1], p2[1], p3[1]) - min(p1[1], p2[1], p2[1], p3[1])) / h, 6)

                if (w_part > 1):
                    w_part = 1
                if (h_part > 1):
                    h_part = 1    

                image_new = cv.cvtColor(image_new, cv.COLOR_BGRA2BGR)
                #image_new = cv.cvtColor(image_new, cv.COLOR_BGR2GRAY)
                if(random.randint(0, 1)):
                    image_new = image_new[int(0.2*h):int(1.2*h), int(0.2*w):int(1*w)]
                    #print(image_new.shape)
                    
                    min_random_edje = 0
                    max_random_edje = int(image_new.shape[1] * random.uniform(0.7, 1))
                    w_start = random.randint(min_random_edje, max_random_edje)
                    w_end = w_start + image_new.shape[0]

                    #print(image_new.shape, w_start,w_end, min_random_edje, max_random_edje)
                    crop_img = image_new[0:image_new.shape[0], w_start:w_end]
                else:
                    crop_img = image_new

                # crop_img = image_new

                width = 256
                height = 256

                dsize = (width, height)
                crop_img = cv.resize(crop_img, dsize)

                crop_img, _ = vp.Image_augmentation().image_horizontal_shift(image = crop_img, ratio = random.uniform(0.01, 0.6)) 
                crop_img, _ = vp.Image_augmentation().image_vertical_shift(image = crop_img, ratio = random.uniform(0.01, 0.6))

                #y_start = 0
                #x_start = 0

                #h_crop = image_new.shape[0]
                #w_crop = h_crop

                #print(image_new.shape, math.floor((image_new.shape[1]/image_new.shape[0] - 1) / 0.2) - 1, w_crop, h_crop)

                #for i in range(math.floor((image_new.shape[1]/image_new.shape[0] - 1) / 0.2) - 1):
                #image_copy = image_new.copy() 
                #crop_img = image_copy[y_start:y_start + h_crop, x_start:x_start + w_crop]
                #print(crop_img.shape, image_new.shape)
                file_creation_time = str(datetime.now())
                    
                #y_start = y_start + h_crop
                #x_start = x_start + w_crop

                #f = open(str(path) + file_creation_time + str('.txt'),"w+")
                #f.write(str(i) + str(" ") + str(center_x) + str(" ") + str(center_y) + str(" ") + str(w_part) + str(" ") + str(h_part))
                #f.close()  


                if j % 10 == 0:
                    place = 'val/'
                else:
                    place = 'train/'

                cv.imwrite(str(output_path) + place + class_img + file_creation_time + str('.JPG'), crop_img)
                counter = counter + 1

def bar_code_detection(video_path = "/data/darknet_AB/python/src/barcode.mp4"):
    input_video = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    c = 0
    err_count = 0

    while input_video.isOpened():
            
        ret, frame = input_video.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray, img_bin = cv.threshold(gray, 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)#[1]
        gray = cv.bitwise_not(img_bin)
        kernel = np.ones((2, 1), np.uint8)
        img = cv.erode(gray, kernel, iterations = 1)
        img = cv.dilate(img, kernel, iterations = 1)
        #gray = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        #gray = cv.medianBlur(gray, 3)
        filename = "/data/darknet_AB/python/src/barcode.jpg"
        #sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=5)
        out_below = pytesseract.image_to_string(img).encode('utf-8')
        print("OUTPUT: ", out_below)
        #cv.imwrite(filename, img) 
        #text = pytesseract.image_to_string(Image.open(filename))
        #os.remove(filename)
        #print(text.encode('utf-8'))
        #img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #print(pytesseract.image_to_string(img_rgb).encode('utf-8'))

        try:
            print (image_to_string(frame))
            err_count = 0
            decoded_objects = pyzbar.decode(frame)
            for obj in decoded_objects:

                # draw the barcode
                print("detected barcode:", obj)
                frame = draw_barcode(obj, frame)
                # print barcode type & data
                print("Type:", obj.type)
                print("Data:", obj.data)

                #    cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                #    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
                #    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            #cv.imwrite(str(image_folder) + str(c) + str(".JPG"), frame)
            #print(" * plus one frame")
            cv.imwrite(str("/data/yolact/toVideo/") + str(c) + str(".JPG"), frame)
            c = c + 1

        except KeyboardInterrupt:
            input_video.release()
            break

        except:
            print("!")
            err_count = err_count + 1
            if (err_count > 3):
                input_video.release()
                break

def color_correction(reference_path, current_path):

    '''def find_color_card(image):

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

        except:
            return None

        cardCoords = np.array([topLeft, topRight,
            bottomRight, bottomLeft])

        card = four_point_transform(image, cardCoords)

        return card


    print(" * Image loading")
    ref = cv.imread(reference_path)
    image = cv.imread(current_path)

    #ref = imutils.resize(ref, width = 600)
    #image = imutils.resize(image, width = 600)


    print(" * Finding color matching cards")
    refCard = find_color_card(ref)
    imageCard = find_color_card(image)

    if refCard is None :
        print(" * Could not find color matching card in refCard :c")
        sys.exit(0)

    if imageCard is None :
        print(" * Could not find color matching card in imageCard :c")
        sys.exit(0)

    print(" * Image matching")
    imageCard = exposure.match_histograms(imageCard, refCard, 
        multichannel = True)

    cv.imwrite("color_cerrection_image.JPG", imageCard)

    # show our input color matching card after histogram matching
    '''
    '''
    src = cv.imread(current_path)
    ref = cv.imread(reference_path)


    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel = multi)
    #matched = vp.Image_augmentation().image_gamma_correction(image = matched, gamma = 0.8)
    #clahe = cv.createCLAHE(clipLimit = 2, tileGridSize = (8, 8))
    #qualized = clahe.apply(equalized)
    #equalized_new = cv.cvtColor(equalized,cv.COLOR_GRAY2RGB)

    cv.imwrite("color_cerrection_image.JPG", matched)'''

    from plantcv import plantcv as pcv
    rgb_img, path, filename = pcv.readimage("target.JPG")
    mask = pcv.transform.create_color_card_mask(rgb_img=img, radius=10, start_coord=(400,600), spacing=(30,30), ncols=6, nrows=4)
    cv.imwrite("target_mask.JPG", mask)

