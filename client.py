#!/usr/bin python3
__title__ = "AutoLabeling"
__version__ = "1.1"
__license__ = "None"
__copyright__ = "None"
__category__  = "Product"
__status__ = "Development"
__author__ = "Ivan Zhezhera"
__company__ = "it-enterprise"
__maintainer__ = "None"
__email__ = "zhezhera@it-enterprise.com"

from datetime import datetime
import socket
import json
import sys
import os

import utils_my as utm



class Client:
    def __init__(self): 
        pass

    def installation():
        os.system("git clone https://github.com/chentinghao/download_google_drive.git")
        os.system("mkdir -p /weights")
        os.system("python3 ./download_google_drive/download_gdrive.py 1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP ./yolact_plus_resnet50_54_800000.pth")
        print(" * Installation is done!")


    def printing(self):
        print("")
        print(" ***** CLient APP v_1.1 *****")
        print("===============================================")
        print("WHAT ACTION DO YOU PREFER?")
        print("===============================================")
        print("choose and push correct button:")
        print(" *","          -- Menu")
        print(" auto_label"," -- Autolabeling")
        print(" ig","         -- Image generation")
        print(" aei","        -- Add flip images")
        print(" t","          -- Train and valid files preparation")
        print(" anchors_gen","-- Anchors generation")
        print(" m_object","   -- Model training")
        print(" m_dek_train","-- Model training")
        print(" m_creation"," -- Keras model creation ")
        print(" m*","         -- Model training (tiny)")
        print(" c","          -- Model training continue")
        print(" c*","         -- Model training continue (tiny)")
        print(" m_convert","  -- Model convertation (tiny)")
        print(" m_obj_test"," -- Test object detection model on single image")
        print(" m_dek_test"," -- Test dekol classification model on single class folder")
        print(" vph","        -- Video creation")
        print(" vid_cutter"," -- Video cutter")  
        print(" bc","         -- Barcode detection")
        print(" h","          -- Hist drawing")
        print(" i","          -- Installation (Yolact only)")
        print(" f","          -- Number correction")
        print(" start","      -- Start on Flask Web Server")
        print(" exit","       -- Closing the Client app")
        print(" calibration","-- Camera calibration")
        print(" rs","         -- Real size")
        print(" rc","         -- Recursive files cleaning by mask")
        print(" color","      -- Color correction")
        print(" gen_yolo","   -- Generation images for yolo")
        print(" gen_yolo_l"," -- Generation images for yolo by scaling clone over yorself")
        print(" gen_deks","   -- Generation images of dekols by augmentation default images")
        print(" cutter","     -- Cutter for images by labels data")
        print(" cutter2","    -- Cutter for images by models prediction")
        print(" cutter3","    -- Cutter for images by user size")
        print(" dd","         -- Drop dublicates in images dataset")
        print(" gc","         -- Get coordinates of boxes")
        print(" bu","         -- Image data build-up (light) for the classifications task")
        print(" af","         -- Add filter for all images in folder")
        print(" acm","        -- Add_color_mask")
        print(" lma","        -- ML model Lime analysis on image")
        
        print("===============================================")
        print(" la","         -- Make list of objects")
        print(" fc","         -- Data preparation: cleaning from blured images, and small objects")
        print(" dh","         -- Data hist")
        print(" oc","         -- Count objects")
        print(" cch","        -- Class changer")
        print(" mrg","        -- Merge *.csv files")
        print(" c2t","        -- csv2txt transformation")
        print(" cnt","        -- Counter of images and labels")
        print(" c2s","        -- Comma 2 space")
        #print(" kmat","  -- KMAT list")
        print("===============================================")
        print("")


    def transmit_data(self, data):
        """ 
        Transmition data method. Had used in request processing part of the program. 
        Method send requests to the server
        input  -> data [string]
        output -> None
        """
        connection = socket.socket()
        connection.connect((settings['connection']['host'] , settings['connection']['tcp_data_port_1']))
        print(" * Connection is done")

        try: 
            if (data != ''):
                connection.sendall(data)
                print(" * Command had sended")
                connection.close ()
        except :
            print(" ! Transmition error")
            connection.close()
            sys.exit(0)


    def reciev_and_transmit_data(self, data):
        connection = socket.socket()
        connection.connect((settings['connection']['host'], settings['connection']['tcp_data_port_1']))
        print(" * Connection is done")
        try: 
            if (data != ''):
                connection.sendall(data)
                print(" * Command had sended")
                #self.connection.close ()

                print(" * Recieving is starting...")
                while 1:
                    try:
                        connection.sendall(b'get')  
                        print("Had sent get msg") 
                        data = connection.recv(1024).decode('utf-8')    
                        print("Must be some msg")                     
                        if (data != ''):
                            print (data)

                        elif (data == ''):
                            pass

                        else:
                            print(" * Data transmition had closed")
                            connection.close()
                            #sys.exit()

                    except KeyboardInterrupt:
                        connection.sendall(b'stop')   

                        print(" ! Custom process canceling")
                        connection.close()

                    except ConnectionError:
                        print(" ! Server connection is dropped")
                        connection.close()
                        #sys.exit()
        except:
            connection.close()
            print(" ! Connection had closed")
            


    def requests(self, action):
        """
        Client starter functiion for images auto labeling.
        Can label '*.JPG' images with and without ARUCO markers.
        """
        if (action == "auto_label" or action == "AUTO_LABEL"):

            answer_1 = input("  Do You want to add marks over the images? Press [Y/N] to continue:  ")
            if (answer_1 == "Y" or answer_1 == "y"):
                print(" Step for images resizing and ARUCA marks adding" )
                os.system("python3 auto_labeling.py -am")
                print(" * Marks have been added")

            else:
                print(" Ok!")
                answer_3 = input(" Remember, next step need images with ARUCO marsk. Press [Enter] to continue")

            print(" Do You want to use smart recognizing algorithms [1] or robust detection algorithm [2] or rubust detection algorithm without aruco [3]" )
            answer_2 = input("  Press [1/2/3] to continue:  ")
            if (answer_2 == "1"):
                os.system("python3 ./eval.py --trained_model=./weights/trained/yolact_resnet50_54_800000.pth --config=yolact_resnet50_config --score_threshold=0.15 --top_k=15 --images=input:output") 
                print(" * All areas have been segmented") 
                os.system("python3 auto_labeling.py -pr")
                print(" * Marks have been found")
                os.system("python3 auto_labeling.py -mg")
                print(" * Final list file has been preparated")
                #os.system("python3 auto_labeling.py -chb") 

            elif(answer_2 == "2"):
                os.system("python3 auto_labeling.py -is")
                print(" * All areas have been segmented") 

            elif (answer_2 == "3"):
                os.system("python3 auto_labeling.py -ril")
                print(" * All areas have been segmented")

            else:
                print(" Uncorrect value!")


        elif(action == "m_obj_test" or action == "M_OBJ_TEST"):
            """
            Method for the yolo model testing
            """
            os.chdir(settings['links']['path_parent'])
            # for the single image
            #os.system("./darknet detector test ./data/names.data ./data/yolov4_sp_10_new.cfg ./data/13.01.22.weights /data/yolact/input/242.JPG")

            # for the folder with images
            utm.model_testing(settings = settings)
            print("[INFO] Done!")

        elif (action == "acm" or action == "ACM"):
            """
            Method for the color mask adding.
            Neede for the yolo model quality work checking.
            """
            utm.add_color_mask(path = settings['links']['input_data_path'])

        elif(action == "af" or action == "AF"):
            """
            Method for the light color filter adding.
            """
            time_start = str(datetime.now())
            utm.add_filter(path = settings['links']['input_data_path']) #"/data/yolact/DEKOL/classification/imds_nano_3/train/2/"
            time_end = str(datetime.now())
            utm.send_report_to_email(message = "Filter adding process has done!", time_start = time_start, time_end = time_end)

        elif(action == "m_creation" or action == "M_CREATION"):
            """
            Method for the model creation for the stack models training.
            """
            path = "/data/yolact/tf_models/stack_models/"
            model_name = (str(datetime.now()) + str(".h5")).replace(" ", "_")

            utm.keras_model_creation(path = path, model_name = model_name, target_size = (200,200), info_flag = True)

        elif(action == "mst" or action == "MST"):
            """ 
            Method of the stack models training
            """
            time_start = str(datetime.now())
            utm.keras_stack_models_training(
                models_path = settings['links']['keras_stack_models_path'], 
                images_path = "", 
                batch_size = settings['NN']['keras_batch_size'], 
                target_size = (200, 200),#tuple(map(int, settings['NN']['keras_target_size'].split(', '))),    # 
                n_epochs = settings['NN']['keras_n_epochs'], 
                verbose = settings['NN']['keras_verbose'],  
                info_flag = settings['NN']['keras_info_flag'])

            time_end = str(datetime.now())
            utm.send_report_to_email(message = "Stack models training had finished!", time_start = time_start, time_end = time_end)
             

        elif(action == "cutter" or action == "CUTTER"):
            """
            Method for the data preparation. Will cut images by label data if yolo format.
            """
            utm.image_cutter_4_classification_labels(
                path_input = settings['links']['input_data_path'], 
                path_output = settings['links']['output_data_path'])

        elif(action == "cutter2" or action == "CUTTER2"):
            """
            Method for cutting by yolo model prediction 
            """
            time_start = str(datetime.now())
            utm.image_cutter_4_classification_by_model_prediction(
                path_input = settings['links']['input_data_path'], 
                path_output = settings['links']['output_data_path'])
            utm.send_report_to_email(message = "Cutter process has done!", time_start = time_start)

        elif(action == "cutter3" or action == "CUTTER3"):
            """
            Method for cutting by static frame size/ 
            """
            time_start = str(datetime.now())
            utm.image_cutter_4_classification_by_size_frame(
                path_input = settings['links']['input_data_path'], 
                path_output = settings['links']['output_data_path']
                )
            utm.send_report_to_email(message = "Cutter process has done!", time_start = time_start)
        
        elif(action == "bu" or action == "BU"):
            """
            Method for image data generation. 
            Will add clones to same folder.
            """
            print("Clonning in:" + str(path))
            answer = input("  Do You want to make clones? Press [Y/N] to continue:  ")
            if (answer == "Y" or answer == "y"):
                utm.image_data_build_up_light(path = settings['links']['input_data_path'])
            
            else:
                pass

        elif(action == "t" or action == "T"):
            """
            Method for data preparation for the yolov4 training.
            Will make train and validation groups of images and their label files.
            """
            #utm.file_list_preparation(self.file_name, self.images_path, self.docker_images_path, self.file_path)
            #utm.file_format_regester_checker(path = self.)
            answer = input("  Do You want to create colored clones for the image DataSet? Press [Y/N] to continue:  ")
            
            utm.folders_cleaning(settings['links']['train_path'], settings['links']['valid_path'])
            print("[INFO] Train and valid folders have cleaned! ")
            if (answer == "Y" or answer == "y"):
                utm.colored_clone()
                print("[INFO] Clones have created! ")
            utm.data_preparation(settings)
            

        elif(action == "m_dek_test" or action == "M_DEK_TEST"):
            """
            Client starter function for the keras model testing method.
            """
            os.chdir(settings['links']['path_parent'])
            utm.CNN_dekol_model_testing(settings = settings) 

        elif(action == "m_dek_train" or action == "M_DEK_TRAIN"):
            """
            Client starter functiion for ML model training.
            You have to choose ...
            """
            #os.chdir()
            result = 1
            model_name = "model_v4_08.03.22.[12].h5"
            #n_epochs = 9

            os.chdir(settings['links']['path_parent'] + "./../yolact/")

            answer = input(" Do You want to use own-[0] arch or default-[1]? Press [0/1] to continue:  ")
            if (answer == "0" or answer == "O" or answer == "o"):
                epoch_quantity = input(" * Choose quantity of epoch: ")
                n_epochs = int(epoch_quantity)
                
                time_start = str(datetime.now())
                result, info = utm.CNN_dekol_model_training(settings = settings, n_epochs = n_epochs, model_name = model_name)
                time_end = str(datetime.now())
                if result == 0:
                    print("[INFO] Model training had finishd.")
                    print("[INFO] Model had saved in dolder './tf_models'. " )
                else:
                    print("[INFO] Model training not finished!")
                body = "model_name:" + model_name + "\nn_epochs:" + str(n_epochs) + "\ninfo:\n" + str(info)
                utm.send_report_to_email(message = "Dekol model training has done! ", time_start = time_start, time_end = time_end, filename = "/data/yolact/tf_models/" + model_name[:-2] + str("png"), body = body)

            elif(answer == "1"):  
                commands_index = input(" * Choose model: \n[0] - VGG11, \n[1] - Resnet18, \n[2] - Mobilenet_v2:, \n[3] - Densenet201 (!), \n[4] - Wide_ResNet-101-2, \n[5] - Resnext101_32x8d (!), \n[6] - Inceptionv3 (!), \n[7] - Resnet152, \n[8] - NASNetLarge (!), \n[9] - ResNet152V2 (!) \n * Model is: ")
                
                bs = settings['NN']['keras_batch_size']
                commands = {'0':'scratch', '1':'finetune', '2':'transfer', '3':'Densenet201', '4':'Wide_ResNet-101-2', '5':'Resnext101_32x8d', '6':'inception_v3', '7':'resnet152', '8':'NASNetLarge', '9':'ResNet152V2'}
                epoch_quantity = input(" * Choose quantity of epoch: ")

                time_start = str(datetime.now())
                try:
                    epoch = int(epoch_quantity)
                    os.system("python3 train_dekol_model.py --mode=" + commands[commands_index] + " --epoch=" + str(epoch_quantity) + " --bs=" + str(bs) + " --train=" + str('./DEKOL/classification/imds_nano_7/train') + " --valid=" + str('./DEKOL/classification/imds_nano_7/val') + " --model=" + str('model_15.02.22_1.pth'))
                    
                except:
                    print(" ! Value of epoch quantity is empty. Will use default value = 5")
                    epoch_quantity = settings['NN']['keras_n_epochs']
                    os.system("python3 train_dekol_model.py --mode=" + commands[commands_index] + " --epoch=" + str(epoch_quantity) + " --bs=" + str(bs) + " --train=" + str('./DEKOL/classification/imds_nano_7/train') + " --valid=" + str('./DEKOL/classification/imds_nano_7/val') + " --model=" + str('model_15.02.22_1.pth'))
                        
                time_end = str(datetime.now())
                
                #print(" * Checking accuracy of 'model_21.08.21_1.pth' on eval dataset in './DEKOL/classification/imds_small_light_l/ev/'" )
                #if(True):
                #os.system("python3 eval.py --dir=eval --model = model_21.08.21_1.pth")
                utm.send_report_to_email(message = "Dekol model training has done!", time_start = time_start, time_end = time_end)


            else:
                print("[INFO] Uncorrect command")

        elif (action == "lma" or action == "LMA"):
            """
            Client starter function for lime analysis.
            """
            os.chdir(settings['links']['path_parent'])
            utm.lime_analysis(settings = settings,
                image = "/data/yolact/DEKOL/classification/imds_nano_9/test/5/2021-10-04 13_30_03.413432.JPG",
                model = "/data/yolact/tf_models/model_v2_22.02.22.[12].h5", 
                report = True)

        elif(action == "m_object" or action == "M_OBJECT"):
            """
            Client starter function for yolov4 training.
            www.localhost:8090 can show progress of training
            """
            time_start = str(datetime.now())
            os.chdir(settings['links']['path_parent'])
            #os.system("./darknet detector train data/names.data data/yolov4_sp_10_new.cfg data/yolov4.conv.137 -dont_show -mjpeg_port 8090 -map") #data/yolov4.conv.137
            #os.system("./darknet detector train data/names.data data/3_models.cfg data/yolov4.conv.137 -dont_show -mjpeg_port 8090 -map") #data/yolov4.conv.137
            os.system("./darknet detector train data/names.data data/densenet201_yolo_3.cfg data/yolov4.conv.137 -dont_show -mjpeg_port 8090 -map") #data/yolov4.conv.137
            #os.system("./darknet detector train data/names.data data/densenet201_yolo.cfg data/yolov4.conv.137 data/densenet201.weights -dont_show -mjpeg_port 8090 -map")
            time_end = str(datetime.now())
            utm.send_report_to_email(message = "Object model training has done!", time_start = time_start, time_end = time_end)

        elif(action == "m*" or action == "M*"):
            """
            Client starter function for yolov4 tiny training.
            www.localhost:8090 can show progress of training
            """
            os.chdir(settings['links']['path_parent'])
            os.system("./darknet detector train data/names.data data/yolov4-tiny-custom.cfg data/yolov4-tiny.conv.29 -dont_show -mjpeg_port 8090 -map")
            #os.system("./darknet detector train data/names.data data/yolov4-tiny-custom.cfg data/yolov4-tiny.conv.29 -dont_show -mjpeg_port 8090 -map")
        
        elif(action == "anchors_gen" or action == "ANCHORS_GEN"):
            """
            Client starter function for anchors generation.
            Needed for yolo config file setup.
            """
            os.chdir(settings['links']['path_parent'])

            classes_quantity = input("  Write quantity af classes, please: ")
            if (int(classes_quantity) < 1000):
                model_size = input("  Write model size, please (can be 416 or 608): ")
                if (int(model_size) == 416 or int(model_size) == 608):
                    os.system("./darknet detector calc_anchors data/names.data -num_of_clusters " + str(classes_quantity) + " -width " + str(model_size) + " -height 416")
                    print("[INFO] Anchors generation is Done!")

            else:
                print("[INFO] Uncorrect INPUT value")
            

        elif (action == "c" or action == "C"):
            """
            Client starter function for the yolov4 training continues method.
            """
            os.chdir(settings['links']['path_parent'])
            os.system("./darknet detector train data/names.data data/3_models.cfg backup/3_models_last.weights -dont_show -mjpeg_port 8090 -map")
            #os.system("./darknet detector train data/names.data data/yolov4_sp_10.cfg backup/yolov4_sp_10_last.weights -dont_show -mjpeg_port 8090 -map")
        
        elif (action == "ig" or action == "IG"):
            """
            Client starter function for the image data generation.
            """
            time_start = str(datetime.now())
            #os.chdir(settings['links']['path_parent'])
            iterations = input(" * Choose quantity of iterations: ")
            utm.dekol_files_auto_preparation(images_path = settings['links']['dekol_png_images'], 
                train_images_path = settings['links']['dekol_train_images_path'], 
                val_images_path = settings['links']['dekol_val_images_path'], 
                eval_images_path = settings['links']['dekol_eval_images_path'], 
                access_rights = 0o755, iterations = iterations)
            time_end = str(datetime.now())
            utm.send_report_to_email(message = "Image generation has done!", time_start = time_start, time_end = time_end)
        
        elif (action == "gen_yolo" or action == "GEN_YOLO"):
            """
            Client starter function for the yolo data augmentation.
            Will make clones from labeled images and copy label files for them.
            """
            time_start = str(datetime.now())
            utm.image_transformation_yolo()
            time_end = str(datetime.now())
            utm.send_report_to_email(message = "Image generation has done!", time_start = time_start, time_end = time_end)

        elif (action == "gen_yolo_l" or action == "GEN_YOLO_L"):
            utm.image_transformation_yolo_light()        

        elif (action == "gen_deks" or action == "GEN_DEKS"):
            """
            Client starter function for the keras image data augmentation.
            """
            utm.image_transformation_old()

        elif (action == "aei" or action == "AEI"):
            """
            Client starter function for manual adding method of extra images to dataset. 
            Add flipped and darkness images.
            Light vetsion of image augmentatiob method.
            Method has option for keras and yolo (with pairs) models.
            """
            # for dekols
            result = input(" Choose format of data generation: without of label files [0], please: ")
            if (result == 0):
                utm.add_extra_images(path = settings['links']['input_data_path'], quantity = 1000)
            
            elif(result == 1):
                # for yolov4 
                utm.add_extra_pairs(path = settings['links']['input_data_path'])

        elif (action == "color" or action == "COLOR"):
            reference_path = './3.JPG'
            current_path = './4.JPG'
            utm.color_correction(reference_path = reference_path, current_path = current_path)
        
        elif (action == "c*" or action == "C*"):
            os.chdir(settings['links']['path_parent'])
            os.system("./darknet detector train data/names.data data/yolov4-tiny-custom_light.cfg backup/yolov4-tiny-custom_light_last.weights -dont_show -mjpeg_port 8090 -map")

        elif(action == "i" or action == "I"):
            """
            exit client app"""
            self.installation()

        elif (action == "dd" or action == "DD"):
            utm.drop_dublicates(path ='/data/yolact/toVideo/')
            print("[INFO] All dublicates had dropped.")

        elif(action == "vph" or action == "VPH"):
            """Video creation util"""
            time_start = str(datetime.now())
            #utm.recurcive_folders_cleaning(path = str(settings['links']['screenshots']), file_format = settings['video']['file_format'])
            #utm.video_writer_posuda_horizontal(video_path = "/data/yolact/labels/003.avi", rotation = 0, settings = settings)
            #utm.fast_video_to_frames(video_path = "/data/yolact/1611/15.12.21.avi", rotation = 0, settings = settings)
            #utm.tiff_to_JPG()
            utm.img_to_video(fps = 5, path = '/data/yolact/input/') #settings['links']['screenshots']
            #utm.send_report_to_email(message = "Video already done!", time_start = time_start )
        
        elif(action == "video_recovery" or action == "VIDEO_RECOVERY"):
            utm.video_recovery_function()

        elif(action == "gc" or action == "GC"):
            time_start = str(datetime.now())
            utm.get_coordinates(path_images_vert = '', 
                path_images_hor = '/data/yolact/worker_morning_13/img/horizontal_2/',
                path_file = './')
            utm.send_report_to_email(message = "File has done!", time_start = time_start )

        elif(action == "bc" or action == "BC"):
            """exit client app"""
            utm.bar_code_detection()


        elif(action == "exit" or action == "EXIT"):
            """exit client app"""
            sys.exit()

        elif(action == "*"):
            self.printing()

        elif(action == "f" or action == "F"):
            utm.file_correction_3(path = "/data/darknet_AB/data/extra/")

        elif(action == "h" or action == "H"):
            print(" Please, putt all images in folder: '/data/yolact/input/'")
            print(" All images must be with *.JPG formt without space in file name")
            os.system("python3 auto_labeling.py -hst") 

        elif (action == "start" or action == "START"):
            msg = "s"
            self.transmit_data(msg.encode('utf-8'))

        elif (action == "calibration" or action == "CALIBRATION"):
            print(" ! *.bmp  images with chess board have to be in ./darknet_AB/python/src/calibration/")
            input("   Press Enter to continue...")
            utm.camera_calibration()

        elif (action == "rs" or action == "RS"):
            utm.real_size()

        elif (action == "m_convert" or action == "M_CONVERT"):
            os.chdir("/data/yolact/")
            print(" ! tiny *.weights file have to be in ./data/")
            input("   Press 'Enter' to continue...")
            #os.system('read -s -n 1 -p "Press any key to continue..."')
            #os.system("python3 save_model.py --weights /data/yolact/data/yolov4-tiny-custom_best.weights --output ./checkpoints/tiny-416_dekol_1 --model yolov4 --tiny")
            os.system("python3 save_model.py --weights /data/yolact/data/yolov4_sp_10_best.weights --output ./checkpoints/yolov4_vertical_608 --model yolov4 --tiny False --input_size 608") #--tiny  --size 608

        elif(action == "vid_cutter" or action == "VID_CUTTER"):
            os.chdir("/data/darknet_AB/python/src/")
            file_name = input("   Write name of file")
            start_time = input("   Set new start time of video in format: hh:mm:ss")
            duration = input("   Set duration of video in format: hh:mm:ss")
            new_file_name = input("   Set new file name of cutted video ")
            os.system("ffmpeg -ss " + str(start_time) + " -i " + str(file_name) + " -t " + str(duration) + " -vcodec copy -acodec copy " + str(new_file_name) )
            print(" * Done! Video is ready!")
            

        elif (action == "la" or action == "LA"):
            utm.labels_analyzer(path = "/data/darknet_AB/data/total_data/")    

        elif(action == "oc" or action == "OC"):
            utm.objects_counters()    
        
        elif(action == "cch" or action == "CCH"):
            try:
                interesting_class = int(input("   Write value of classs: "))
                utm.find_interesting_classes(path = "/data/darknet_AB/data/TOTAL_VERTICAL/[5]_pan/", 
                    path_to_save = "/data/darknet_AB/data/total_data/[4]_kettle_35/", class_number = interesting_class)
            
            except:
                print("[INFO] Uncorrect input value")

        elif(action == "mrg" or action == "MRG"):
            utm.merger_of_csv()

        elif(action == "rc" or action == "RC"):
            utm.recurcive_folders_cleaning(path = './video_recorder/', file_format = '.avi')
        #elif (action == "kmat" or action == "KMAT"):
        #    pass

        elif(action == "c2t" or action == "C2T"):
            utm.csv2txt_transformation()


        elif(action == "cnt" or action == "CNT"):
            utm.counter()


        elif(action == "c2s" or action == "C2S"):
            utm.comma2space()

        elif(action == "dp" or action == "DP"):
            fm = utm.blured_images_detection()

        elif(action == "dh" or action == "DH"):
            path = "/data/darknet_AB/data/data_classes/" 
            path_to_save = "/data/darknet_AB/data/hist_of_images.pdf"

            utm.data_hist(path = path, path_to_save = path_to_save)
            print("[INFO] Hist report has done at: ", path_to_save)

        elif(action == "fc" or action == "FC"):
            path = '/data/darknet_AB/data/data_classes/'
            input(" ! Will be cleaned in folder: " + str(path) + ". Press 'Enter' to continue...")
            utm.files_cleaning(path = path, size_level = 0.03, blur_level = 110)

        else:
            print("Uncorrect command. Try again")

            


if __name__ == "__main__": 
    """
    Starter for client part (main loop-function).
    Make requests for the server part or operations with data.
    """
    global settings

    with open("./settings.json", "r") as read_file:
        settings = json.load(read_file)
    
    CL =  Client()
    
    while True: 
        try:
            action = input(">> ")
            CL.requests(action) 

        except ConnectionError:
            print("[INFO] Check connection!")

        except KeyboardInterrupt:
            print("[INFO] Closing")
            sys.exit(0)
