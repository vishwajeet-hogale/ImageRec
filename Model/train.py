import time
import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,model_from_json
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
# from extractfaces import extractfaces as ef
import extractfaces as ef

BATCH_SIZE = 32
image_count = 68
TRAIN_STEPS_PER_EPOCH = np.ceil((image_count*0.8/BATCH_SIZE)-1)
VAL_STEPS_PER_EPOCH = np.ceil((image_count*0.2/BATCH_SIZE)-1)
TrainingImagePath='../DataAugmentation/Image/Final Training Images'
TestImagePath='../DataAugmentation/Image/Final Testing Images'
LOOP_DIR = "../DataAugmentation/Image/Final Training Images/Gaurav"
Face_dict = {}


def make_dataset(BATCH_SIZE,TRAIN_STEPS_PER_EPOCH,VAL_STEPS_PER_EPOCHS):
        train_datagen = ImageDataGenerator(
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True)
        
        # Defining pre-processing transformations on raw images of testing data
        # No transformations are done on the testing images
        test_datagen = ImageDataGenerator()
        
        # Generating the Training Data
        training_set = train_datagen.flow_from_directory(
                TrainingImagePath,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')
        
        
        # Generating the Testing Data
        test_set = test_datagen.flow_from_directory(
                TestImagePath,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')

        # class_indices have the numeric tag for each face
        TrainClasses=training_set.class_indices
        
        # Storing the face and the numeric tag for future reference
        ResultMap={}
        for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
            ResultMap[faceValue]=faceName
        
        # Saving the face map for future reference
        
        with open("ResultsMap.pkl", 'wb') as fileWriteStream:
            pickle.dump(ResultMap, fileWriteStream)
        
        # The model will give answer as a numeric tag
        # This mapping will help to get the corresponding face name for it
        print("Mapping of Face and its ID",ResultMap)
        
        # The number of neurons for the output layer is equal to the number of faces
        OutputNeurons=len(ResultMap)
        print('\n The Number of output neurons: ', OutputNeurons)
        return training_set,test_set,OutputNeurons,ResultMap

def train_on_folder(training_set,TRAIN_STEPS_PER_EPOCH,VAL_STEPS_PER_EPOCH,OutputNeurons):
        classifier= Sequential()
        classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
        classifier.add(MaxPool2D(pool_size=(2,2)))
        classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        classifier.add(MaxPool2D(pool_size=(2,2)))
        classifier.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        classifier.add(MaxPool2D(pool_size=(2,2)))
        classifier.add(Flatten())
        classifier.add(Dense(64, activation='relu'))
        classifier.add(Dense(OutputNeurons, activation='softmax'))
        classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
        
        # Measuring the time taken by the model to train
        StartTime=time.time()

        # Starting the model training
        classifier.fit_generator(
                            training_set,
                            steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                            epochs=35,
                            validation_data=test_set,
                            validation_steps = VAL_STEPS_PER_EPOCH)
        
        EndTime=time.time()
        print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')
        return classifier

def test(classifier,ResultMap,ImagePath='../DataAugmentation/Image/Final Testing Images/Gaurav/3face2.jpg'):
        test_image = tf.keras.utils.load_img(ImagePath,target_size=(64, 64))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=classifier.predict(test_image,verbose=0)
        print('####'*10)
        print('Prediction is: ',ResultMap[np.argmax(result)])
        return ResultMap[np.argmax(result)]
def loop_dir_extract_faces(classifier,ResultMap,path = LOOP_DIR):
    for i in os.listdir(path):
        if(i.endswith(".jpg")):
            for ind,img in enumerate(ef.get_faces(path + "/" + i)):
                #     print(img)
                    cv2.imwrite("./Output/img.jpg",img)
                    prediction = test(classifier,ResultMap,"./Output/img.jpg")
                    print(prediction)
                    if prediction in Face_dict:
                        Face_dict[prediction].append(path + "/" + i)
                    else:
                        Face_dict[prediction] = [path + "/" + i]
if __name__ == "__main__":
        train = False
        Test = False
        classifier = None

        if train :
                training_set,test_set,OutputNeurons,ResultMap = make_dataset(BATCH_SIZE,TRAIN_STEPS_PER_EPOCH,VAL_STEPS_PER_EPOCH)
                classifier = train_on_folder(training_set,TRAIN_STEPS_PER_EPOCH,VAL_STEPS_PER_EPOCH,OutputNeurons)
                # serialize model to JSON
                model_json = classifier.to_json()
                with open("./models/model.json", "w") as json_file:
                        json_file.write(model_json)
                # serialize weights to HDF5
                classifier.save_weights("./models/model.h5")
                print("Saved model to disk")


        # Test
        json_file = open('./models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        # load weights into new model
        classifier.load_weights("./models/model.h5")
        print("Loaded model from disk")
        ResultMap = pickle.load(open('ResultsMap.pkl', 'rb'))
        if Test:
                test(classifier,ResultMap)

        loop_dir_extract_faces(classifier,ResultMap)
        print(Face_dict)
        os.remove("./Output/img.jpg")
        
