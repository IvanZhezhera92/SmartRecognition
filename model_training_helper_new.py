# ПРИМЕР РЕАЛИЗАЦИИ
#https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_flow_from_directory.ipynb

# ОПИСАНИЕ СЛОЕВ
#https://www.geeksforgeeks.org/python-image-classification-using-keras/

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam

batch_size = 128
target_size = (200, 200)
n_epochs = 24

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale = 1 / 255, 
                                   rotation_range = 60,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   horizontal_flip = True
                                   )

val_datagen = ImageDataGenerator(rescale = 1 / 255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './DEKOL/classification/imds_nano_3/train/',  # This is the source directory for training images
        target_size = target_size,  # All images will be resized to 200 x 200
        batch_size = batch_size,
        # Specify the classes explicitly
        classes = ['0','1', '2', '3', '4'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode = 'categorical')

val_generator = val_datagen.flow_from_directory(
        './DEKOL/classification/imds_nano_3/val/',  # This is the source directory for training images
        target_size = target_size,  # All images will be resized to 200 x 200
        batch_size = batch_size,
        # Specify the classes explicitly
        classes = ['0','1', '2', '3', '4'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode = 'categorical')


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color

    # The first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The sixth convolution
    #tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    #tf.keras.layers.MaxPooling2D(2, 2),

    # The seventh convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    #tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation = 'relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5, activation = 'softmax') #'softmax'
])



'''model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation = 'relu'),
    #tf.keras.layers.Dropout(0.5),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5, activation = 'softmax')
])'''


model.summary()
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = 0.001), #rho = 0.9, epsilon = 1e-07, centered = True, #RMSprop
              metrics = ['acc'])

total_sample = train_generator.n

#history = model.fit(
#        train_generator,
#        steps_per_epoch = int(total_sample / batch_size),  
#        epochs = n_epochs,
        
#        validation_data = val_generator,
#        validation_steps = int(int(total_sample / batch_size)),
#        verbose = 1)

#model_name = './tf_models/model_v2_26.11.21.[5].h5'
#model.save(model_name)
#print(model_name + " had saved!")


'''def train_CNN(train_directory,target_size=(200,200), classes=None,
              batch_size=128,num_epochs=20,num_classes=5,verbose=0):
    """
    Trains a conv net for the flowers dataset with a 5-class classifiction output
    Also provides suitable arguments for extending it to other similar apps
    
    Arguments:
            train_directory: The directory where the training images are stored in separate folders.
                            These folders should be named as per the classes.
            target_size: Target size for the training images. A tuple e.g. (200,200)
            classes: A Python list with the classes 
            batch_size: Batch size for training
            num_epochs: Number of epochs for training
            num_classes: Number of output classes to consider
            verbose: Verbosity level of the training, passed on to the `fit_generator` method
    Returns:
            A trained conv net model
    
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import tensorflow as tf
    from tensorflow.keras.optimizers import RMSprop
    
    # ImageDataGenerator object instance with scaling
    train_datagen = ImageDataGenerator(rescale=1/255,
                                    #rotation_range=40,
                                    #width_shift_range=0.2,
                                    #height_shift_range=0.2,
                                    #shear_range=0.2,
                                    #zoom_range=0.2,
                                    #horizontal_flip=True,
                                    #fill_mode='nearest')

    # Flow training images in batches using the generator
    train_generator = train_datagen.flow_from_directory(
            train_directory,  # This is the source directory for training images
            target_size=target_size,  # All images will be resized to 200 x 200
            batch_size=batch_size,
            # Specify the classes explicitly
            classes = classes,
            # Since we use categorical_crossentropy loss, we need categorical labels
            class_mode='categorical')
    
    input_shape = tuple(list(target_size)+[3])
    
    # Model architecture
    model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 512 neuron in the fully-connected layer
    tf.keras.layers.Dense(512, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Optimizer and compilation
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
    
    # Total sample count
    total_sample=train_generator.n
    
    # Training
    model.fit(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=num_epochs,
        verbose=verbose)
    
    return model'''


#import cv2 as cv
#frame_path = str("/data/TOTAL/153_2.JPG")
#image_default = cv.imread(frame_path)
#resized = cv.resize(image_default, (200,200), interpolation = cv.INTER_AREA)


from PIL import Image
import numpy as np
import fnmatch, os
import video_processing as vp

#model_name = 'model_v1_04.10.21.[3].h5'
model_name = './tf_models/model_v1_29.10.21.[5].h5'
#model_name = './tf_models/model_v2_26.11.21.[5].h5'

savedModel = load_model(model_name)
#path = "./../TOTAL/"

path = "/data/yolact/DEKOL/classification/imds_nano_3/data_from_progon/3/" #/data/yolact/DEKOL/classification/imds_nano_3/val/0/
images = fnmatch.filter(os.listdir(path), '*.JPG')

print(len(images))

counter = 0

for i in range(len(images)):
    try:
        #image_default = Image.open(path + "48_1.JPG")
        image_default = Image.open(path + images[i])
        
        #pil_width, pil_height = image_default.size
        #image_default = image_default.crop((int(pil_width * 0.01), int(pil_height * 0.3), int(pil_width * 0.99), int(pil_height * 0.99)))

        image_default = image_default.resize(target_size)
        image_default = np.expand_dims(image_default, axis = 0)
        cv_image_default = np.array(image_default)
        print(cv_image_default)
        #print(vp.Video_Processing().polka_detection(image = cv_image_default))
        
        #print(images[i], str(" -> "), np.around(savedModel.predict(image_default)[3], decimals = 1))

        
        #print(savedModel.predict(image_default))
        #if(np.around(savedModel.predict(image_default)[0], decimals = 1)[2] == 1):
        #    counter = counter + 1

    except Exception as e: 
        print(e)
        #print(" ! Some error")


print("\tResult: ", round(counter / len(images) * 100, 2))

#41.49
#100 100
#77.67 99.03
#89.47 94.74



