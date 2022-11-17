# importing libraries
import cv2
import numpy as np
import face_recognition as face_rec

# function
def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width,height)
    return cv2.resize(img,dimension,interpolation= cv2.INTER_AREA)

# img declaration
imgElon = face_rec.load_image_file('imagesbasic/elon_musk.jpg')
imgElon = resize(imgElon,0.50)
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElon_test = face_rec.load_image_file('imagesbasic/bill_gates.jpg')
imgElon_test = resize(imgElon_test,0.50)
imgElon_test = cv2.cvtColor(imgElon_test, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_Elon = face_rec.face_locations(imgElon)[0]
encode_Elon = face_rec.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLocation_Elon[3],faceLocation_Elon[0]),(faceLocation_Elon[1],faceLocation_Elon[2]),(255, 0, 255),3)

faceLocation_Elontest = face_rec.face_locations(imgElon_test)[0]
encode_Elontest = face_rec.face_encodings(imgElon_test)[0]
cv2.rectangle(imgElon_test, (faceLocation_Elon[3],faceLocation_Elon[0]),(faceLocation_Elon[1],faceLocation_Elon[2]),(255, 0, 255),3)

results = face_rec.compare_faces([encode_Elon],encode_Elontest)  # comparing images
print(results)
cv2.putText(imgElon_test,f'{results}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('main_img',imgElon)
cv2.imshow('test_img',imgElon_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
