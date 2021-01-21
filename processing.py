import cv2
import sys
from imutils import paths
import numpy as np
import imutils

# Les constantes 

KNOWN_DISTANCE = 12.0
KNOWN_WIDTH = 12.0
kernel = np.ones((3,3),'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (0,20)  
fontScale = 0.6
color = (0, 0, 255)
thickness = 2
#LES FONCTIONS NECESAIRES
def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    # trouver les contours dans l'image bordée et conserver le plus grand;
    # nous supposerons qu'il s'agit de notre feuille de papier dans l'image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)
   # calculer la boîte englobante de la région papier et la renvoyer
    return cv2.minAreaRect(c) 
def distance_to_camera(knownWidth, focalLength, perWidth):
    # calculer et renvoyer la distance entre le fabricant et la caméra
    return (knownWidth * focalLength) / perWidth
def message_accueil(face):
      cv2.putText(face, "SALUT TU VEUX JOUER AVEC MOI?", org, font, 1, color, 2, cv2.LINE_AA)

        
#PRINCIPALE

cap=cv2.VideoCapture(0) 
faceCascarde=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

while(True): 
    rep,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascarde.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=4,
                                        minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),4)
        if len(faces)>0:
            for face in faces :
                message_accueil(face)
    image=cv2.imread('visage.jpg')
    marker = find_marker(image)
    focalLength = (marker[1][0] * KNOWN_DISTANCE)/KNOWN_WIDTH
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    
    image1 = cv2.putText(gray, 'Distance de la caméra en CM :', org, font,  
                          1, color, 2, cv2.LINE_AA)
    image = cv2.putText(image1, str(inches/12), (110,50), font,  fontScale, color, 1, cv2.LINE_AA)

    cv2.imshow('image',image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows() 
cap.release() 