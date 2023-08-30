import cv2
import numpy as np 
import face_recognition

# imggaurav = face_recognition.load_image_file('C:/Users/GAURAV/Desktop/Python Projects/Photos/gaurav.jpg')
# imggaurav = cv2.cvtColor(imggaurav.cv2.COLOR_BGR2RGB)

imgaryan = face_recognition.load_image_file('C:/Users/GAURAV/Desktop/Python Projects/Photos/aryan.jpg')
imgaryan = cv2.cvtColor(imgaryan.cv2.COLOR_BGR2RGB)

imgmohit = face_recognition.load_image_file('C:/Users/GAURAV/Desktop/Python Projects/Photos/mohitimgmohit.jpg')
imgmohit = cv2.cvtColor(imgmohit.cv2.COLOR_BGR2RGB)



# cv2.imshow('Gaurav suthar', imggaurav)
cv2.imshow('Aryan Verma', imgaryan)
cv2.imshow('Mohit Kumawat', imgmohit)


cv2.waitKey(0)
