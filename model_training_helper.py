#https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_flow_from_directory.ipynb

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
import fnmatch, os, re
from PIL import Image
import numpy as np

def train_CNN(image_train_directory = './DEKOL/classification/imds_nano_5/train/', 
    target_size = (200,200), batch_size = 32, 
    n_epochs = 11, summary = True, verbose = True,
    model_path = "./tf_models/",
    model_name = 'model_v1_19.01.22.[12].h5'):
    """
    Method for CNN model training for dekol classification
    input     -> image_train_directory [train folder with classificated subfolders]
              -> target_size [size of input matrix]
              -> batch_size [batch_size]
              -> n_epochs [quantity of iteration]
              -> summary [writing information about model]
              -> verbose [writing progress bar of process]
              -> model_path [path to model folder]
              -> model_name [name of model file]
    output    -> --
    """
    folders_list = os.listdir(image_train_directory)
    folders_list.sort(key = lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    num_classes = len(folders_list)

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale = 1/255, rotation_range = 15,
        width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.1,
        zoom_range = 0.1, horizontal_flip = True) #,fill_mode = 'nearest'

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            image_train_directory,  # This is the source directory for training images
            target_size = target_size,  # All images will be resized to 200 x 200
            batch_size = batch_size,
            # Specify the classes explicitly
            classes = folders_list,
            # Since we use categorical_crossentropy loss, we need categorical labels
            class_mode = 'categorical')

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
        # The first convolution
        tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (target_size[0], target_size[1], 3)),
        tf.keras.layers.MaxPooling2D((2, 2), strides = 2),
        # The second convolution
        
        tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),
        tf.keras.layers.MaxPooling2D((2,2), strides = 2),

        tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),
        tf.keras.layers.MaxPooling2D((2,2), strides = 2),
        
        # Flatten the results to feed into a dense layer
        tf.keras.layers.Flatten(),
        # 256 neuron in the fully-connected layer
        tf.keras.layers.Dense(256, activation = 'relu'),
        # 5 output neurons for 5 classes with the softmax activation
        tf.keras.layers.Dense(num_classes, activation = 'softmax')
    ])
    
    if (summary): model.summary()

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 0.001),
                  metrics = ['acc'])

    total_sample = train_generator.n

    history = model.fit(
            train_generator, 
            steps_per_epoch = int(total_sample / batch_size),  
            epochs = n_epochs,
            verbose = verbose)
    
    model.save(model_path + model_name)
    print("[INFO] " + model_name + " had saved in: ", model_path, " folder")





#import cv2 as cv
#frame_path = str("/data/TOTAL/153_2.JPG")
#image_default = cv.imread(frame_path)
#resized = cv.resize(image_default, (200,200), interpolation = cv.INTER_AREA)

def test_CNN(model_name = 'model_v1_19.01.22.[12].h5',
    path = "/data/yolact/DEKOL/classification/imds_nano_5/val/0/"):
    
    savedModel = load_model(model_name)
    
    images = fnmatch.filter(os.listdir(path), '*.JPG')
    print(len(images))
    counter = 0
    for i in range(len(images)):
        try:
            #image_default = Image.open(path + "48_1.JPG")
            image_default = Image.open(path + images[i])

            image_default = image_default.resize((200,200))
            image_default = np.expand_dims(image_default, axis = 0)

            print(images[i], str(" -> "), np.around(savedModel.predict(image_default)[0], decimals = 1))
            print("index: ",np.around(savedModel.predict(image_default)[0], decimals = 1).index(1))
            #print(savedModel.predict(image_default))
            if(np.around(savedModel.predict(image_default)[0], decimals = 1)[4] == 1):
                counter = counter + 1

        except:
            print(" ! Some error")

    print(counter / len(images) * 100)

from PIL import Image
import numpy as np
import fnmatch, os

path = "/data/yolact/DEKOL/classification/imds_nano_5/val/8/"
image_name = "Shooting 1_0056_6tbvfjdn7osrtgf.JPG.JPG"
model_name = '/data/yolact/tf_models/model_v1_19.01.22.[12].h5'

savedModel = load_model(model_name) 

image_default = Image.open(path + image_name)

image_default = image_default.resize((200,200))
image_default = np.expand_dims(image_default, axis = 0)

print(image_name, str(" -> "), np.around(savedModel.predict(image_default)[0], decimals = 1))


if(np.around(savedModel.predict(image_default)[0], decimals = 1)[8] == 1):
    print("\nYupii!")

else:
    print("Shit happens")

prediction = np.around(savedModel.predict(image_default)[0], decimals = 1)
[float(i) for i in prediction]  
print("index", prediction.tolist().index(1))



#train_CNN()