import cv2

def get_faces(image_path='../DataAugmentation/Image/Final Testing Images/Gaurav/gk_aug10.jpg'):
    image = cv2.imread(image_path)
    print(image)
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
    cv2.destroyAllWindows()
    return all_faces