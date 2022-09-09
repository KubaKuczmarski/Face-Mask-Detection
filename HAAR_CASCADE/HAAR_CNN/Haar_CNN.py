# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:16:26 2022

Subject: Face mask detection using Haar cascade fac clasificator and Transfer Learining neural network.

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek
import numpy as np
import tensorflow as tf
import cv2
import time

# Wczytanie modelu
model = tf.keras.models.load_model('MobileNet/face_mask_detection_MobilNet.h5')

# Rozmiar zdjęcia
img_size = 150

# Klasyfikator Haara
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ustawienia kamery
cap = cv2.VideoCapture(0)
cap.set(3,640) # ustawienie szerokoci
cap.set(4,480) # ustawienie wysokoci

# Dane potrzebne do wyznaczenia ilosci FPS
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyswietlanych napisów
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    
    _, img = cap.read()
    frame_id += 1
    
    # Załadowanie obrazu wejciowego w skali szroci
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Definicja parametrów klasyfikatora Haara
    faces = faceCascade.detectMultiScale(
        gray_img,        # Wejsciowy obrazek w skali szarosci.
        scaleFactor=1.2, # Parametr określający stopień zmniejszenia rozmiaru obrazu w każdej skali obrazu. Służy do tworzenia piramidy skali.
        minNeighbors=5,  # Parametr określający, ilu sąsiadów powinien mieć każdy kandydujący prostokąt, aby go zachować. 
                         # Wyższa liczba daje mniej przypadków false positives.   
        minSize=(20,20) # Minimalny rozmiar prostokąta, który należy uznać za twarz.
    )
    
    # Zaznacenie za pomocą prostokąta twarzy (wykryjanie i zaznaczanie twarzy na obrazie)
    for (x,y,w,h) in faces:
        
        # Wycięte zdjęcia twarzy z prostokątów
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
        
        facess = faceCascade.detectMultiScale(roi_gray)

        # Obróbka wyciętych zdjęć twarzy
        final_image = cv2.resize(roi_color, (img_size,img_size))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image/255.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Predykcja - 0:'INCORRECT MASK', 1:'FACE MASK', 2:'NO MASK'
        predictions = model.predict(final_image)
        
        # Wybór największej wartoci z tabeli i okrelenie, na ktorym miejscu się znajduje
        label = np.argmax(predictions, axis=1)

        # Zmiana koloru prostokąta i wyswietlenie tekstu w zależnoci od predykcji
        if label == 0:
            status = "INCORRECT MASK"
            #Narysowanie tła w kształcie prostokąta, na ktorym znajdować się będzie informacja o maseczce
            cv2.rectangle(img, (x-1,y+h), (x + w + 1, y + h + 50), (86,228,253), -1 )
            # Dodanie tekstu 
            cv2.putText(img, status, (x+5, y + h + 30), font, 0.63, (255,255,255), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (86,228,253), 2)
        
        elif label == 1:
            status = "FACE MASK"
            cv2.rectangle(img, (x,y+h), (x + w, y + h + 50 ), (0,255,0), -1 )
            cv2.putText(img, status, (x+5, y + h + 30), font, 0.7, (255,255,255), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            
        elif label == 2:
            status = "NO MASK"
            cv2.rectangle(img, (x-1,y+h), (x + w + 1, y + h + 50 ), (0,0,255), -1 )
            cv2.putText(img, status, (x+30, y + h + 30), font, 0.7, (255,255,255), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
    
    # Liczba FPS
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4)         
            
    # Wywietlenie wyników
    cv2.imshow('FACE MASK DETECTION',img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # zakończenie transmidji przez nacinięcie klawisza 'ESC'
        break
cap.release()
cv2.destroyAllWindows()


