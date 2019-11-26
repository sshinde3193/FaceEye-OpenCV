import cv2
import sys
import os

cascadePath = r'/home/sidd/scripts/venv/lib/python3.6/site-packages/cv2/data/' # replace with yours
face_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_eye.xml')
basePath = r'IMDB-Celeb-Faces'
files = os.listdir(basePath)

for f in files:
    imgPath = os.path.join(basePath,f)
    image = cv2.imread(imgPath) # read
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # conver to grayscale

    ## Making a copy of Image
    image2 = image.copy()

    ## Detecting & converting face to gray scale -> crop to original Image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_crop = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        shp = face_crop.shape
        cv2.imshow('Detected Face', image)
        cv2.imshow('Cropped Face', face_crop)
        image2[y:y+h, x:x+w] = face_crop[:,:,None]
        cv2.imshow('Merged Face' , image2)

    ## Detecting & blurring eyes -> crop to last Image
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y,w,h) in eyes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        eye_crop = cv2.GaussianBlur(image[y:y+h, x:x+w],(5,5),cv2.BORDER_DEFAULT)
        image2[y:y+h, x:x+w] = eye_crop[:,:,:]
        cv2.imshow('Detected eyes', image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
