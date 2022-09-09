# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:27:34 2022

Subject: Face mask detection using dlib face detector and Transfer Learining neural network.

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek (ENG. Loading libraries)
import cv2
import numpy as np
import dlib
import tensorflow as tf
import time

# Wczytanie modelu (ENG. Loading the best  Transfer Learning model)
model = tf.keras.models.load_model('MobileNet/face_mask_detection_MobilNet.h5') 

# Ustawienie rozmiaru zdjęcia (ENG. Set the size of the photo)
img_size = 150

# Wybór kamery (ENG. Camera selection - laptop camera)
cap = cv2.VideoCapture(0)

# Dane potrzebne do wyznaczenia ilości FPS (ENG. Data needed to determine the amount of FPS )
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyświetlanych napisów (ENG. Setting the font of the displayed subtitles)
font = cv2.FONT_HERSHEY_SIMPLEX

# Wczytanie detektora twarzy z biblioteki dlib (ENG. Loading the face detector from the dlib library)
detector = dlib.get_frontal_face_detector()

while True:
    # Uruchomienie kamery (ENG. Start the camera)
    _, img = cap.read()
    frame_id += 1
    
    # Konwersja do odcieni szarości - łatwiejsza obróbka, mniejsze obciążenie CPU (ENG. Convert to grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detekcja twarzy (ENG. Face detection)
    faces = detector(gray)
    for face in faces:
        
        # Wyciągnięcie pojedyńczych punktów ze współrzędnych okręślających położenie twarzy (ENG. Take out single points from the coordinates defining the position of the face)
        # Współrzędne (x,y) górnego lewego punktu (ENG. Coordinates (x, y) of the upper left point)
        x1 = face.left() 
        y1 = face.top()
       
        # Współrzedne (x,y) dolnego prawego punktu (ENG. Coordinates (x, y) of the lower right point)
        x2 = face.right()
        y2 = face.bottom()
         
        # Wycięcie zdjęć twarzy (ENG. Cut out face photos)
        roi_gray = gray[y1:y2, x1:x2]
        roi_color = img[y1:y2, x1:x2]
        
        # Obróbka wyciętych zdjęć w celu poddania ich na wejście do wczytanego modelu sieci neuronowej (ENG. Processing of cut photos and give them to input into the loaded neural network model)
        final_image = cv2.resize(roi_color, (img_size,img_size))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image/255.0
        
        # Predykcji położenia maseczki na twarzy (ENG. Prediction of the position of the mask on the face)
        predictions = model.predict(final_image)
        
        # Wybór największej wartości z przewidzianych (ENG. Selecting the highest value from those provided)
        label = np.argmax(predictions, axis=1)

        # Wybór rodzaju położenia maseczki na twarzy (ENG.Choosing the type of position of the mask on the face)
        if label == 0:
            cv2.rectangle(img, (x1,y1), (x2,y2), (86,228,253), 2)
            cv2.rectangle(img, (x1-1,y2), (x2+1, y2 + 50), (86,228,253), -1 )
            cv2.putText(img, "INCORRECT MASK", (x1, y2+30), font, 0.55, (255,255,255), 2)
        elif label == 1:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1-1,y2), (x2+1, y2 + 50), (0,255,0), -1 )
            cv2.putText(img, "MASKA", (x1+20, y2+30), font, 0.8, (255,255,255), 2)
        elif label == 2:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1-1,y2), (x2+1, y2 + 50), (0,0,255), -1 )
            cv2.putText(img, "NO MASK", (x1+10, y2+30), font, 0.8, (255,255,255), 2)
    
    # Liczba FPS (ENG. FPS number)
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4) 
           
    # Wyświetlenie okna z widokiem z kamery i wynikami (ENG. Display a window with the camera view and results)
    cv2.imshow("FACE MASK DETECTION", img)
    
    # Zakończenie pracy programu (ENG. Ending the program)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Zakończenie transmisji przez naciśnięcie klawisza 'ESC' (ENG. Ending the transmission by pressing the 'ESC' key)
        break
cap.release()
cv2.destroyAllWindows()