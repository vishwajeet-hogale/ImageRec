import pickle
from keras.models import model_from_json

def load_model():
    json_file = open('../Model/models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("../Model/models/model.h5")
    ResultMap = pickle.load(open('../Model/ResultsMap.pkl', 'rb'))
    return classifier,ResultMap

def predict_image(classifier,ResultMap,ImagePath='../DataAugmentation/Image/Final Testing Images/Gaurav/3face2.jpg'):
        test_image = tf.keras.utils.load_img(ImagePath,target_size=(64, 64))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=classifier.predict(test_image,verbose=0)
        print('####'*10)
        print('Prediction is: ',ResultMap[np.argmax(result)])
        return ResultMap[np.argmax(result)]
    