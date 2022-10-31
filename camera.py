#!/usr/bin python3

__author__ = "Ivan Zhezhera"
__company__ = "it-enterprise"
__copyright__ = "None"
__license__ = "None"
__version__ = "1.0"
__maintainer__ = "None"
__email__ = "zhezhera@it-enterprise.com"
__category__  = "Product"
__status__ = "Development"

from flask import Flask, render_template, Response, send_from_directory
from imutils.video import FPS, VideoStream
from pypylon import pylon, genicam
import numpy as np
import cv2 as cv
import random
import os
from datetime import datetime
import time
import tensorflow as tf
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import InteractiveSession, ConfigProto
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import pandas as pd
import sys
import codecs
#from queue import Queue
from collections import deque
from websocket import create_connection

import video_processing as vp
import json
import base64

from skimage.exposure import is_low_contrast


class VideoCamera(object):
    def __init__(self):
        #self.video = cv.VideoCapture("/data/yolact/src/demo_6.MP4")
        #connection
        #self.ws_info = create_connection("ws://172.16.32.73:1880/ws/info")
        #self.ws_camera1 = create_connection("ws://172.16.32.73:1880/ws/camera1")
        #self.ws_camera2 = create_connection("ws://172.16.32.73:1880/ws/camera2")
        #self.ws_camera3 = create_connection("ws://172.16.32.73:1880/ws/camera3")
        #self.ws_camera4 = create_connection("ws://172.16.32.73:1880/ws/camera4")
        
        #LABELS_FILE = "/data/darknet_AB/python/src/coco.names"
        #CONFIG_FILE = "/data/darknet_AB/python/src/config.cfg"
        #WEIGHTS_FILE = "/data/darknet_AB/python/src/29.03.21_1.weights"
        #CONFIDENCE_THRESHOLD = 0.25
        #DATA_FILE = "/data/darknet_AB/python/src/names.data"
        # Files links
        self.LABELS_FILE = './src/coco.names'
        self.CONFIG_FILE = './src/config.cfg' #config-tiny.cfg
        self.WEIGHTS_FILE =  './src/29.03.21_1.weights' 
        self.CONFIDENCE_THRESHOLD = 0.10 
        self.DATA_FILE = './src/names.data'

        # NET configuration
        self.net = cv.dnn_DetectionModel(self.CONFIG_FILE, self.WEIGHTS_FILE)
        self.net.setInputSize(608, 608)  # <- FOR YOLOV4

        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        with open(self.LABELS_FILE, 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')
        # Camera configuration
        #self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        #self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
        self.multicameras_flag = 0
        #==========================================================================
        self.maxCamerasToUse = 3
        self.cameras = pylon.InstantCameraArray(min(len(pylon.TlFactory.GetInstance().EnumerateDevices()), self.maxCamerasToUse))

        self.l = self.cameras.GetSize()
          
        for i, cam in enumerate(self.cameras):
            cam.Attach(pylon.TlFactory.GetInstance().CreateDevice(pylon.TlFactory.GetInstance().EnumerateDevices()[i]))
        
        self.cameras.StartGrabbing()
        self.exitCode = 0
        self.multicamera = 0
        self.encoder_vertical = 0 
        self.tracker_vertical = 0
        self.checker_vertical = 0 
        self.infer_vertical = 0
        self.cameraMatrix_vertical = 0 
        self.distCoeffs_vertical = 0
        self.saved_model_loaded_vertical = 0

        self.encoder_angular = 0 
        self.tracker_angular = 0
        self.checker_angular = 0 
        self.infer_angular = 0
        self.cameraMatrix_angular = 0 
        self.distCoeffs_angular = 0

        self.saved_model_loaded_angular = 0
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config = self.config)
        #==========================================================================
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


        self.ds_factor = 0.4
        self.ds_factor_web = 0.4
        self.deep_init_vertical_flag = 0
        self.deep_init_angular_flag = 0

        self.max_cosine_distance = 0.41
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        self.batch_size = 1
        self.metric_type = "cosine"
        self.model_filename = './model_data/mars-small128.pb'
        self.yolov4_tiny_to_tf_converted_model =  "./checkpoints/dekol" # "./checkpoints/tiny-416_new" #
        self.serving_type = 'serving_default'
        self.max_output_size_per_class = 50
        self.max_total_size = 25
        self.iou_threshold = 0.30
        self.score_threshold = 0.5
        
        #self.model_filename = './checkpoints/tiny-416/saved_model.pb'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(self.physical_devices) > 0:
            tf.config.experimental.set_memory_growth(self.physical_devices[0], True)

        self.kmat_list = 'kmat_list.csv'
        self.frame_0 = None
        self.frame_1 = None
        self.frame_2 = None
        self.frame_3 = None
        # Video codec
        self.FourCC = 'XVID'#'MJPG'
        # Full frame size
        self.video_frame_size = (1942,2590) 
        self.calibrate_image = cv.imread('etalon.JPG')
        self.net = 0
        self.exposure_min_level = 0.45

  
    def get_kmat_data(self):
        df = pd.read_csv(self.kmat_list, encoding='utf-8') #sep = "@", lineterminator='#', 
        new_header = df.iloc[0] #grab the first row for the header
        df = df[1:]             #take the data less the header row
        df.columns = new_header #set the header row as the df header
        print (df) 


        '''def remapping(self, image):
        # undistort
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        dst = cv.undistort(image, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        pass'''

    def get_frame_angular_pylon(self):

        """Transmit video from pylon camera to Flask"""
        prvs_gray = None
        checker, cameraMatrix, distCoeffs = vp.Video_Processing().file_checker()

        while self.camera.IsGrabbing():
            #try:
            if (1):
                #ret = True
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():

                    with open(self.LABELS_FILE, 'rt') as f:
                        names = f.read().rstrip('\n').split('\n')

                    frame = self.converter.Convert(grabResult).GetArray()

                    frame = cv.resize(frame, None, fx = self.ds_factor,fy = self.ds_factor, interpolation = cv.INTER_AREA) #

                    classes, confidences, boxes = self.net.detect(frame, confThreshold = self.CONFIDENCE_THRESHOLD, nmsThreshold = 0.4)
                    print(classes, boxes,confidences)

                    try:
                        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                            frame = cv.rectangle(frame, box, color = (0, 255, 0), thickness = 3)
                            #frame = cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    except:
                        pass

                    '''current_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)  

                    hsv = np.zeros_like(frame)
                    hsv[...,1] = 255

                    #gray = cv.GaussianBlur(gray, (13, 13), 0)
                    if prvs_gray is None:
                        prvs_gray = current_gray
                        continue

                    contours,v = vp.Video_Processing().motion_detection(current_gray, prvs_gray, hsv)
                    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
                    dense_flow = cv.addWeighted(frame, 1, bgr, 2, 0)

                    for contour in contours:
                        if (cv.contourArea(contour) < 55000 or cv.contourArea(contour) > 75000): 
                            continue

                        w, h = vp.Video_Processing().motion_indication(contour, dense_flow)
                        #print(vp.Video_Processing().real_size(checker, current_gray, cameraMatrix, distCoeffs, diameter = h))
                        #print(round(random.uniform(178., 181.),2))

                    frame = dense_flow'''
                ret, jpeg = cv.imencode('.jpg', frame)
                frame_res = jpeg.tobytes() 

                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_res + b'\r\n')
                #prvs_gray = current_gray


            grabResult.Release()
        self.camera.StopGrabbing()




    def deep_parameters_initialization(self):
        # DEEPSORT
        # https://github.com/theAIGuysCode/yolov4-deepsort
        #self.get_kmat_data()
        encoder = gdet.create_box_encoder(self.model_filename, batch_size = 1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        tracker = Tracker(metric)
        #config = ConfigProto()
        #config.gpu_options.allow_growth = True
        #session = InteractiveSession(config = config)
        #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        return encoder, tracker

    def deep_processing(self, frame, encoder, tracker, infer):

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
            max_output_size_per_class = self.max_output_size_per_class,
            max_total_size = self.max_total_size,
            iou_threshold = self.iou_threshold,
            score_threshold = self.score_threshold
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
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
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
            cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv.putText(frame, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255), 2)
            #print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
            track_ids.append(track.track_id)
            boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]) 
        fps = round(1.0 / (time.time() - start_time),2)
        #print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        #result = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        #return result, boxes, fps, str(track.track_id), class_name, classes, track_ids
        return result, boxes, fps, classes, track_ids



    def get_frame_vertical_pylon(self, settings):

        if (self.deep_init_vertical_flag == 0):
            encoder, tracker = self.deep_parameters_initialization()
            saved_model_loaded_vertical = tf.saved_model.load(self.yolov4_tiny_to_tf_converted_model, tags = [tag_constants.SERVING])
            infer = saved_model_loaded_vertical.signatures['serving_default']
            self.deep_init_vertical_flag = 1
            checker, cameraMatrix, distCoeffs = vp.Video_Processing().file_checker()

        while self.camera.IsGrabbing():
            #try:
            if(1):
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():

                    frame = self.converter.Convert(grabResult).GetArray()

                    frame = cv.resize(frame, None, fx = self.ds_factor, fy = self.ds_factor, interpolation = cv.INTER_AREA)                           
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    # marks detection & drawing
                    try:
                        corners, ids = vp.Video_Processing().marks_detection(checker, gray, cameraMatrix, distCoeffs)

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

                    # object detection 
                    #try:
                    #    #print(" * ")
                    result, boxes, fps, classes, track_ids = self.deep_processing(frame = result, encoder = encoder, tracker = tracker, infer = infer)
                                        #except:
                    #    boxes, fps, track_id, class_name, classes, track_ids = None, None, None, None, None, None


                    if boxes:
                        centers = []
                        for i in range(len(boxes)):
                            x = int(abs(boxes[i][0] + boxes[i][2]) / 2)
                            y = int(abs(boxes[i][1] + boxes[i][3]) / 2)
                            centers.append([x, y])

                            result = vp.Drawing().draw_center(result, x, y, radius = 5, color = (0, 0, 255), thickness = -1)

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
                                        #result = vp.Drawing().draw_center(image = result, x = int(abs(center[0])), color = (0,0,0),y = int(abs(center[1])), radius = 10)
                                        print(res)

                                        try:
                                            #self.message(boxes = boxes[i], fps = fps, class_name = classes[i], track_id = track_ids[i], centers = centers[i])
                                            print("Here must be message for")

                                        except:
                                            print("Some error at: ", datetime.now().strftime("%H:%M:%S"))

                                else:
                                    print(" ! Not enought marks! Chek camera position or marks condition")

                    else:
                        pass
                        #print(" ! Some oter case")

                    result = vp.Video_Processing().visual_object_orientation(image = result, corners = corners, ids = ids) #

                    ret, jpeg = cv.imencode('.jpg', result)

                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


 
    def angular_pylon_processing(self, frame):
        if (self.deep_init_angular_flag == 0):
            self.encoder_angular, self.tracker_angular = self.deep_parameters_initialization()
            self.saved_model_loaded_angular = tf.saved_model.load("./checkpoints/yolov4-608", tags = [tag_constants.SERVING])
            self.infer_angular = self.saved_model_loaded_angular.signatures['serving_default']
            self.deep_init_angular_flag = 1
            self.checker_angular, self.cameraMatrix_angular, self.distCoeffs_angular = vp.Video_Processing().file_checker()
            #print(self.checker, self.cameraMatrix, self.distCoeffs)
            print(" * Initialisation of angular camera parameters is complited!")

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        try:
            corners, ids = vp.Video_Processing().marks_detection(self.checker_angular, gray, self.cameraMatrix_angular, self.distCoeffs_angular)
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
            result, boxes, fps, classes, track_ids = self.deep_processing(frame = result, encoder = self.encoder_angular, tracker = self.tracker_angular, infer = self.infer_angular)    
            self.message(cam_index = "angular", boxes = boxes, fps = fps, classes = classes, track_id = track_ids)
        except:
            boxes, fps, classes, track_ids = None, None, None, None       
            self.message(cam_index = "angular",boxes = 0, fps = 0, classes = 0, track_id = 0)
        result = vp.Video_Processing().visual_object_orientation(image = result, corners = corners, ids = ids) 
        return result

    '''def vertical_pylon_processing(self, frame, settings):
        if (self.deep_init_vertical_flag == 0):
            self.deep_init_vertical_flag = 1
            self.checker_vertical, self.cameraMatrix_vertical, self.distCoeffs_vertical = vp.Video_Processing().file_checker()
    
            print(" * Initialisation of vertical camera parameters is complited!")

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        try:
            corners, ids = vp.Video_Processing().marks_detection(self.checker_vertical, gray, self.cameraMatrix_vertical, self.distCoeffs_vertical)
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
            classes, confidences, boxes = net.detect(frame, confThreshold = float(settings['NN']['confidence_threshold']), nmsThreshold = float(settings['NN']['nms_threshold']))
            
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                
                if (float(label) >= float(settings['NN']['confidence_threshold'])):
                    label = '%s: %s' % (self.names[classId], label)

                left_main, top_main, width_main, height_main = box
                top_main = max(top_main, labelSize[1])
                cv.rectangle(frame, box, color = (50, 250, 20), thickness = 4)

        return frame'''


    def vertical_pylon_processing(self, frame):
        if (self.deep_init_vertical_flag == 0):
            self.encoder_vertical, self.tracker_vertical = self.deep_parameters_initialization()
            self.saved_model_loaded_vertical = tf.saved_model.load("./checkpoints/tiny-416", tags = [tag_constants.SERVING])
            self.infer_vertical = self.saved_model_loaded_vertical.signatures['serving_default']
            self.deep_init_vertical_flag = 1
            self.checker_vertical, self.cameraMatrix_vertical, self.distCoeffs_vertical = vp.Video_Processing().file_checker()
            print(" * Initialisation of vertical camera parameters is complited!")

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        try:
            corners, ids = vp.Video_Processing().marks_detection(self.checker_vertical, gray, self.cameraMatrix_vertical, self.distCoeffs_vertical)
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
            result, boxes, fps, classes, track_ids = self.deep_processing(frame = result, encoder = self.encoder_vertical, tracker = self.tracker_vertical, infer = self.infer_vertical)  
            '''with open(self.LABELS_FILE, "rt") as f:
                names = f.read().rstrip("\n").split("\n")

            classes, confidences, boxes = net.detect(result, confThreshold = self.CONFIDENCE_THRESHOLD, nmsThreshold = 0.4)

            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                if (float(label) >= 0.5):
                    #    label = round(random.uniform(0.71, 0.95), 2)
                    label = '%s: %s' % (names[classId], label)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    left, top, width, height = box
                    top = max(top, labelSize[1])
                                
                    cv.rectangle(result, box, color=(0, 255, 0), thickness=3)
                    cv.rectangle(result, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
                    cv.putText(result, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))  
                    fps = None
                    track_ids = None'''
            
        except:
            boxes, fps, classes, track_ids = None, None, None, None    
            
        self.message(cam_index = "vertical", boxes = boxes, fps = fps, classes = str(classes), track_id = track_ids, kmat = None, diam_x = None, diam_y = None, current_time = datetime.now().strftime("%H:%M:$S.$f").rstrip('0'))


        if boxes:
            centers = []
            for i in range(len(boxes)):
                x = int(abs(boxes[i][0] + boxes[i][2]) / 2)
                y = int(abs(boxes[i][1] + boxes[i][3]) / 2)
                centers.append([x, y])
                result = vp.Drawing().draw_center(result, x, y, radius = 5, color = (0, 0, 255), thickness = -1)
            
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
                            #result = vp.Video_Processing().draw_center(image = result, x = int(abs(center[0])), color = (0,0,0),y = int(abs(center[1])), radius = 10)
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

        #_, result_picture = cv.imencode('.png', result) 
        #picture = base64.b64encode(result_picture)
        #self.ws_camera1.send(str(picture))

        return result 
    
    def message(self, cam_index = None, boxes = None, fps = None, focus_level = None, exposure = None, classes = None, track_id = None, kmat = None, diam_x = None, diam_y = None, current_time = None, color = None, dekol = None, err = None):
        message = {}
        message["cam"] = cam_index
        message["fps"] = fps
        message["focus_level"] = focus_level
        message["exposure"] = exposure
        message["class"] = classes
        message["color"] = color
        message["dekol"] = dekol
        message["id"] = track_id
        message["boxes"] = boxes
        message["kmat"] = kmat
        message["diam_x"] = diam_x
        message["diam_y"] = diam_y
        message["time"] = current_time
        message["err"] = err
        #print(message)
        #self.ws_info.send(json.dumps(message))

    #def video_transmition(self, image = None):
    #    self.ws_camera1.send(image.getvalue().encode("base64"))

     

    def get_all_frames_pylon(self, settings, video_recorder = 1, duration = 25, queue_max_len = 48, freq = 14): 
        """ All video streams processing method.
        Main process loop for objects detections and calculations

        input  -> settings        - from settings.json [from serber.py] 
               -> video_recorder  - settings for video writing
               -> duration        - duration is seconds of each video file
               -> queue_max_len   - quantit yof files per camera
               -> freq            - FPS [needed for video writing]
        output -> jpeg.tobytes() string with all cameras frames per one moment
        """    

        if (self.multicamera == 0):
            #initialization
            fourcc = cv.VideoWriter_fourcc(*self.FourCC)
            local_dt_start_vertical = datetime.now()
            file_name_vertical = str(local_dt_start_vertical) + '.avi'
            local_dt_current_vertical = 0

            local_dt_start_horizontal = datetime.now()
            file_name_horizontal = str(local_dt_start_horizontal) + '.avi'
            local_dt_current_horizontal = 0

            local_dt_start_angular = datetime.now()
            file_name_angular = str(local_dt_start_angular) + '.avi'
            local_dt_current_angular = 0

            local_dt_start_horizontal_2 = datetime.now()
            file_name_horizontal_2 = str(local_dt_start_horizontal_2) + '.avi'
            local_dt_current_horizontal_2 = 0

            out_vertical     = cv.VideoWriter('./video_recorder/vertical/'     + file_name_vertical,     fourcc, freq, self.video_frame_size)
            out_horizontal   = cv.VideoWriter('./video_recorder/horizontal/'   + file_name_horizontal,   fourcc, freq, self.video_frame_size)
            out_angular      = cv.VideoWriter('./video_recorder/angular/'      + file_name_angular,      fourcc, freq, self.video_frame_size) #(1036,777)s
            out_horizontal_2 = cv.VideoWriter('./video_recorder/horizontal_2/' + file_name_horizontal_2, fourcc, freq, self.video_frame_size) #(1036,777)s

            self.multicamera = 1

            q_vertical     = deque()
            q_horizontal   = deque()
            q_angular      = deque()
            q_horizontal_2 = deque()

        while self.cameras.IsGrabbing():
            try:     
                while self.cameras.IsGrabbing():
                    
                    self.cameras.Gain = "Continuous"
                    self.cameras.ExposureTime = 10
                    self.cameras.CenterX = True
                    self.cameras.CenterY = True
                    
                    grabResult = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    
                    image = self.converter.Convert(grabResult).GetArray()
                    cameraContextValue = grabResult.GetCameraContext()
                    frame = cv.resize(image, None, fx = self.ds_factor, fy = self.ds_factor, interpolation = cv.INTER_AREA) 
                    h,w,_ = frame.shape

                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    scaled_w = int(gray.shape[1] * self.ds_factor_web)
                    scaled_h = int(gray.shape[0] * self.ds_factor_web)
                    
                    #=================================================================================================================================================================
                    
                    if (cameraContextValue == 0):
                        # VERTICAL
                        self.frame_0 = frame
                        self.frame_0 = self.vertical_pylon_processing(frame = self.frame_0)
                        #self.frame_0 = vp.BarCode().Recognition(image = self.frame_0)
                        # Setting and marking place for the focus 
                        gray = gray[0 : frame.shape[0], int(frame.shape[1] * .25) : int(frame.shape[1] * .7)]
                        #cv.rectangle(self.frame_0, (int(self.frame_0.shape[1] * .25), 0),(int(self.frame_0.shape[1] * .7), int(self.frame_0.shape[0])), color = (255,255,0), thickness = 5)
                        
                        # Level of focus
                        fm = round(cv.Laplacian(gray, cv.CV_64F).var(), 2)
                        cv.putText(self.frame_0, str(round(fm, 1)), (int(w * .75), int(h * .1)), cv.FONT_HERSHEY_SIMPLEX, 2, (209, 80, 0, 255), 3, cv.LINE_AA)
                        
                        if is_low_contrast(gray, fraction_threshold = self.exposure_min_level):
                            cv.putText(self.frame_0, "normal exposure" , (int(w * .75), int(h * .12)), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
                        else:
                            cv.putText(self.frame_0, "low exposure" , (int(w * .75), int(h * .12)), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv.LINE_AA)
                        
                        # Transmition to Edje
                        frame2web = cv.resize(self.frame_0, (scaled_w, scaled_h), interpolation = cv.INTER_AREA)
                        #self.ws_camera1.send(str(base64.b64encode(cv.imencode('.png', frame2web)[1])))

                        
                        if (video_recorder == 1):
                            local_dt_current_vertical = datetime.now()
                            difference_vertical = int((local_dt_current_vertical - local_dt_start_vertical).total_seconds())

                            if (difference_vertical < duration):  
                                cv.putText(self.frame_0, str(datetime.now()), (int(w * .6), int(h*.95)), cv.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3, cv.LINE_AA)                              
                                out_vertical.write(frame)
                                print("add frame")
                                
                            elif(difference_vertical >= duration):
                                out_vertical.release()
                                print(">>> new video")
                                q_vertical.append(file_name_vertical)
                                #print(len(q_vertical), q_vertical)
                                
                                if (len(q_vertical) == queue_max_len):
                                    os.remove(str("./video_recorder/vertical/") + str(q_vertical.popleft())) 
                                    
                                file_name_vertical = str(datetime.now()) + '.avi'
                                out_vertical = cv.VideoWriter('./video_recorder/vertical/' + file_name_vertical, fourcc, freq, self.video_frame_size)
                                local_dt_start_vertical = datetime.now()


                    
                    elif (cameraContextValue == 1):
                        # HORIZONTAL
                        self.frame_1 = frame
                        
                        # Setting and marking place for the focus
                        gray = gray[int(frame.shape[0] * .4) : int(frame.shape[0] * .72), int(frame.shape[1] * .2) : int(frame.shape[1] * .8)]
                        cv.rectangle(self.frame_1, (int(frame.shape[1] * .2), int(frame.shape[0] * .4)),(int(frame.shape[1] * .8), int(frame.shape[0] * .72)), color = (255,255,0), thickness = 5)

                        # Level of focus
                        fm = round(cv.Laplacian(gray, cv.CV_64F).var(), 2)
                        cv.putText(self.frame_1, str(round(fm, 1)), (int(w * .75), int(h * .1)), cv.FONT_HERSHEY_SIMPLEX, 2, (209, 80, 0, 255), 3, cv.LINE_AA)
                        
                        # Transmition to Edje
                        frame2web = cv.resize(self.frame_1, (scaled_w, scaled_h), interpolation = cv.INTER_AREA)
                        #self.ws_camera2.send(str(base64.b64encode(cv.imencode('.png', frame2web)[1])))

                        if (video_recorder == 1):

                            local_dt_current_horizontal = datetime.now()
                            difference_horizontal = int((local_dt_current_horizontal - local_dt_start_horizontal).total_seconds())

                            if (difference_horizontal < duration):  
                                cv.putText(self.frame_1, str(datetime.now()), (int(w * .6), int(h*.95)), cv.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3, cv.LINE_AA)                              
                                out_horizontal.write(self.frame_1)

                                
                            elif(difference_horizontal >= duration):
                                out_horizontal.release()
                                q_horizontal.append(file_name_horizontal)
                                #print(len(q_horizontal), q_horizontal)
                                
                                if (len(q_horizontal) == queue_max_len):
                                    os.remove(str("./video_recorder/horizontal/") + str(q_horizontal.popleft())) 
                                    
                                file_name_horizontal = str(datetime.now()) + '.avi'
                                out_horizontal = cv.VideoWriter('./video_recorder/horizontal/' + file_name_horizontal, fourcc, freq, self.video_frame_size)
                                local_dt_start_horizontal = datetime.now()


                    elif (cameraContextValue == 2): 
                        # ANGULAR
                        self.frame_2 = frame
                        self.frame_2 = cv.rotate(self.frame_2, cv.ROTATE_180)
                        '''try:
                            classes, confidences, boxes = self.net.detect(self.frame_2, confThreshold = self.CONFIDENCE_THRESHOLD, nmsThreshold = 0.4)

                            self.message(cam_index = "angular", boxes = boxes, fps = None, classes =  classes, diam_x = None, diam_y = None, kmat = None, current_time = datetime.now().strftime("%H:%M:%S.%f").rstrip('0'),track_id = None, err = None) #  
                            
                            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                                #print(classId, confidence, box)  
                                left, top, width, height = box  
                                label = '%.2f' % confidence
                                label = '%s: %s' % (self.names[classId], label)
                                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                top = max(top, labelSize[1])
                                cv.rectangle(self.frame_2, box, color = (0, 255, 0), thickness = 3)
                                cv.rectangle(self.frame_2, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
                                cv.putText(self.frame_2, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
                           

                        except:
                            pass'''
                        #self.frame_2 = self.angular_pylon_processing(frame = self.frame_2)
                        
                        # Setting and marking place for the focus
                        gray = gray[int(self.frame_2.shape[0] * .5) : int(self.frame_2.shape[0]), int(self.frame_2.shape[1] * .3) : int(self.frame_2.shape[1] * .8)]
                        cv.rectangle(self.frame_2, (int(self.frame_2.shape[1] * .3), int(self.frame_2.shape[0] * .5)),(int(self.frame_2.shape[1] * .8), int(self.frame_2.shape[0])), color = (255,255,0), thickness = 5)

                        # Level of focus
                        fm = round(cv.Laplacian(gray, cv.CV_64F).var(), 2)
                        cv.putText(self.frame_2, str(round(fm, 1)), (int(w * .75), int(h * .1)), cv.FONT_HERSHEY_SIMPLEX, 2, (209, 80, 0, 255), 3, cv.LINE_AA)
                        
                        # Transmition to Edje
                        frame2web = cv.resize(self.frame_2, (scaled_w, scaled_h), interpolation = cv.INTER_AREA)
                        #self.ws_camera3.send(str(base64.b64encode(cv.imencode('.png', frame2web)[1])))

                        if (video_recorder == 1):
                            local_dt_current_angular = datetime.now()
                            difference_angular = int((local_dt_current_angular - local_dt_start_angular).total_seconds())

                            if (difference_angular < duration):  
                                cv.putText(self.frame_2, str(datetime.now()), (int(w * .6), int(h*.95)), cv.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3, cv.LINE_AA)                              
                                out_angular.write(self.frame_2)
                                
                            elif(difference_angular >= duration):
                                out_angular.release()
                                q_angular.append(file_name_angular)
                                #print(len(q_horizontal), q_horizontal)
                                
                                if (len(q_angular) == queue_max_len):
                                    os.remove(str("./video_recorder/angular/") + str(q_angular.popleft())) 
                                    
                                file_name_angular = str(datetime.now()) + '.avi'
                                out_angular = cv.VideoWriter('./video_recorder/angular/' + file_name_angular, fourcc, freq, self.video_frame_size)
                                local_dt_start_angular = datetime.now()



                    elif (cameraContextValue == 3):
                        # HORIZONTAL_2
                        self.frame_3 = frame

                        if (video_recorder == 1):
                            local_dt_current_horizontal_2 = datetime.now()
                            difference_horizontal_2 = int((local_dt_horizontal_2 - local_dt_start_horizontal_2).total_seconds())

                            if (difference_horizontal_2 < duration):  
                                cv.putText(self.frame_0, str(datetime.now()), (int(w * .6), int(h*.95)), cv.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3, cv.LINE_AA)                              
                                out_horizontal_2.write(self.frame_3)
                                
                                
                            elif(difference_horizontal_2 >= duration):
                                out_horizontal_2.release()
                                print(">>> new video")
                                q_horizontal_2.append(file_name_horizontal_2)
                                #print(len(q_vertical), q_vertical)
                                
                                if (len(q_horizontal_2) == queue_max_len):
                                    os.remove(str("./video_recorder/horizontal_2/") + str(q_horizontal_2.popleft())) 
                                    
                                file_name_horizontal_2 = str(datetime.now()) + '.avi'
                                out_horizontal_2 = cv.VideoWriter('./video_recorder/horizontal_2/' + file_name_horizontal_2, fourcc, freq, self.video_frame_size)
                                local_dt_start_horizontal_2 = datetime.now()


                    
                    frames = [self.frame_0, self.frame_1, self.frame_2, self.frame_3]

                    if any(x is None for x in frames):
                        for i in range(len(frames)):
                            if frames[i] is None:
                                frames[i] = np.zeros((h, w, 3), np.uint8)

                    vis = np.concatenate((frames[0], frames[1], frames[2]), self.frame_3, axis = 0)
                    #vis = cv.cvtColor(vis, cv.COLOR_GRAY2RGB)
                    ret, jpeg = cv.imencode('.jpg', vis)

                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            except GeneratorExit:
                sys.exit(1)


            except KeyboardInterrupt:
                print(" ! Close all")
                sys.exit(1)

            except:

                pass



    def videoToWeb(self, host, tcp_flask_port, settings, mode = 0):
        checker, cameraMatrix, distCoeffs = vp.Video_Processing().file_checker()

        if (mode == 0):

            app = Flask(__name__)

            def shutdown_server():
                func = request.environ.get('werkzeug.server.shutdown')
                if func is None:
                    raise RuntimeError('Not running with the Werkzeug Server')
                func() 

            @app.route('/')
            def index():
                # rendering webpage
                return render_template('index.html')

            @app.route('/favicon.ico') 
            def favicon(): 
                return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

            @app.route('/video_feed')
            def video_feed():
                return Response(self.get_all_frames_pylon(settings), mimetype = 'multipart/x-mixed-replace; boundary=frame')
                #return Response(main(net, path_parent, stream_scale_factor, mode, conn, settings, 0), mimetype='multipart/x-mixed-replace; boundary=frame')

            @app.route('/shutdown', methods=['POST'])
            def shutdown():
                print(" ! STOP!!!!!!!")
                shutdown_server()
                return ' ! Server shutting down...'

            app.debug = False
            app.config['TEMPLATES_AUTO_RELOAD'] = True
            app.run(host, tcp_flask_port, threaded = True)