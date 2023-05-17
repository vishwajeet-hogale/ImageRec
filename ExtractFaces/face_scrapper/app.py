# Importing the libraries.
import cv2
import sys

# Here we will pass the image as an argument during the runtime and that is done like this.
#imagepath = sys.argv[1]

image = cv2.imread('C:/Users/NIDHSHE/face_scrapper/people.png') #Converts the image into OpenCV object.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Here the image into gray scale.

# Identifying the faces from group image.
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
) 

print("Found {0} Faces!".format(len(faces)))

# Draw rectangle around each faces.
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi_color = image[y:y + h, x:x + w] # selecting the faces from the rectangle.
    print("Found the objects and Saving them locally")
    cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)# saving the faces locally.

status = cv2.imwrite('faces_detected.jpg', image) # writing the rectangled faces to local filesystem.
print("Writing faces_detected.jpg to filesystem was: ",status) 