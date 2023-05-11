import os
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image
datagen = ImageDataGenerator(        
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))
path =r'Image/'
#we shall store all the file names in this list
filelist = []
for root, dirs, files in os.walk(path):
	for file in files:
        #append the file name to the list
		filelist.append(os.path.join(root,file))
SIZE = 224
#my_images = os.listdir(filelist)
for i, image_name in enumerate(filelist):
    
    dataset = []
    ###image_name.split('\\')
    if (image_name.split('.')[1] == 'jpg'):
        save_to_dir=image_name.split('\\')[0]
        print(save_to_dir)  

        image = io.imread(image_name) 
        image = Image.fromarray(image, 'RGB')        
        image = image.resize((SIZE,SIZE)) 
        dataset.append(np.array(image))
        x = np.array(dataset)
        i = 0
        for batch in datagen.flow(x, batch_size=16,
                          save_to_dir= save_to_dir,
                          save_prefix='dr',
                          save_format='jpg'):
                          i += 1    
                          if i > 50:
                              break



















#print all the file names
for name in filelist:
   print(name)