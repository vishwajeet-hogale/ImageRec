import os
import shutil
import pickle
from keras.models import model_from_json
import numpy as np
import cv2
import tensorflow as tf
import time
import pickle

LOOP_DIR = "./DataAugmentation/group_image"
OUTPUT_DIR = "./Output"
# source_dir = ""
def copy_files(name,image_list):
        source_dir = './DataAugmentation/group_image'
        destination_dir = f'./Output/{name}'
        try:
                os.mkdir(destination_dir)
        except:
                pass
        # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']

        # Iterate over the list of images
        for image_file in image_list:
        # Construct the source and destination file paths
                source_file = source_dir + "/" + image_file
                # timestamp = str(int(time.time()))
                destination_file = destination_dir + "/" + image_file
                # Use shutil.copy2() to copy the image file
                # try:
                shutil.copy2(source_file, destination_file)
                # except:
                        # continue
def get_faces(image_path='../DataAugmentation/Image/Final Testing Images/Gaurav/gk_aug10.jpg'):
    image = cv2.imread(image_path)
    # print(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    all_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        # Display or save the extracted face image
        # cv2.imshow('Face', face)
        all_faces.append(face)
        # print(face)
        # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    return all_faces
def load_model():
    json_file = open('./Model/models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("./Model/models/model.h5")
    ResultMap = pickle.load(open('./Model/ResultsMap.pkl', 'rb'))
    return classifier,ResultMap
def get_pred(result,ResultMap):
        result = result.tolist()
        preds = [float(round(i,8)) for i in result[0]]
        print(preds)
        new_res = max(preds)
        if new_res < 0.50:
                return "Unknown"
        return ResultMap[preds.index(new_res)]
def predict_image(classifier,ResultMap,ImagePath='../DataAugmentation/Image/Final Testing Images/Gaurav/3face2.jpg'):
        test_image = tf.keras.utils.load_img(ImagePath,target_size=(64, 64))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=classifier.predict(test_image,verbose=0)
        print('####'*10)
        print('Prediction is: ',ResultMap[np.argmax(result)])
        print("Result : ",result)
        # return ResultMap[np.argmax(result)]
        return get_pred(result,ResultMap)
def loop_dir_extract_faces(classifier,ResultMap,path = LOOP_DIR):
        Face_dict = {}
        for i in os.listdir(path):
                if(i.endswith(".jpg") or i.endswith(".jpeg")):
                        for ind,img in enumerate(get_faces(path + "/" + i)):
                                print(i)
                                cv2.imwrite("./Output/img.jpg",img)
                                prediction = predict_image(classifier,ResultMap,"./Output/img.jpg")
                                print(prediction)
                                if prediction in Face_dict:
                                        Face_dict[prediction].add(path + "/" + i)
                                else:
                                        Face_dict[prediction] = {path + "/" + i}
        return Face_dict

def train():
        classifier,ResultMap = load_model()
        Face_dict = loop_dir_extract_faces(classifier,ResultMap,LOOP_DIR)
        return Face_dict

# print(ef)
def write_face_dict(FaceDict,path):
        with open(path + "/" + "FaceDict.pkl","wb") as f:
                pickle.dump(FaceDict,f)

def load_face_dict(path):
        file_name = path + "/" + "FaceDict.pkl"
        FaceDict = pickle.load(open(file_name, 'rb'))
        return FaceDict
        
                