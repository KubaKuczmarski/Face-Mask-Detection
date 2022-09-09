# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:27:34 2022

Subject: Face mask detection using dlib face detector and Transfer Learining neural network.

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek
import cv2
import numpy as np
import dlib
import tensorflow as tf
import time

# Wczytanie modelu
model = tf.keras.models.load_model('MobileNet/face_mask_detection_MobilNet.h5') #VGG19/face_mask_detection_VGG19.h5

#Rozmiar zdjęcia
img_size = 150

# Wybór kamery
cap = cv2.VideoCapture(0)

# Dane potrzebne do wyznaczenia ilosci FPS
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyswietlanych napisów
font = cv2.FONT_HERSHEY_SIMPLEX

# Wczytanie detektora twarzy z biblioteki dlib
detector = dlib.get_frontal_face_detector()

while True:
    # Uruchomienie kamery
    _, img = cap.read()
    frame_id += 1
    
    # Konwersja do odcieni szrosci - łatwiejsza obróbka, mniejsze obciązenie CPU
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detekcja twarzy
    faces = detector(gray)
    for face in faces:
        
        # Wyciągamy pojedyńcze punkty ze wspołrzędnych okręslajacych połozenie twarzy 
        # wspołrzędne (x,y) górnego lewego punktu
        x1 = face.left() 
        y1 = face.top()
        # wspołrzedne (x,y) dolnego prawgo punktu
        x2 = face.right()
        y2 = face.bottom()
         
        # Wycięcie zdjęć twarzy
        roi_gray = gray[y1:y2, x1:x2]
        roi_color = img[y1:y2, x1:x2]
        
        # Obróbka wyciętych zdjęć w celu podania ich na wejscie do wczytanego modelu sieci neuronowej
        final_image = cv2.resize(roi_color, (img_size,img_size))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image/255.0
        
        # Dokonanie predykcji połozenia maeczki na twarzy
        predictions = model.predict(final_image)
        
        # Wybór największej warosci z przewidzianych
        label = np.argmax(predictions, axis=1)

        # Wybór przewidywanej opcji położenia maseczki na twarzy
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
    
    # Liczba FPS
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4) 
           
    # Wywietlenie okna z widokiem z kamery i wynikami
    cv2.imshow("FACE MASK DETECTION", img)
    
    # Zakonczenie pracy programu
    k = cv2.waitKey(30) & 0xff
    if k == 27: # zakończenie transmidji przez nacinięcie klawisza 'ESC'
        break
cap.release()
cv2.destroyAllWindows()