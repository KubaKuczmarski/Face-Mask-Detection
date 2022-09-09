# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:16:26 2022

Subject: Face mask detection using Haar cascade fac clasificator and Transfer Learining neural network.

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek (ENG. Loading libraries)
import numpy as np
import tensorflow as tf
import cv2
import time

# Wczytanie modelu (ENG. Loading the best Transfer Learning model)
model = tf.keras.models.load_model('MobileNet/face_mask_detection_MobilNet.h5')

# Ustawienie rozmiaru zdjęcia (ENG. Set the size of the photo)
img_size = 150

# Klasyfikator Haara - twarz (ENG. Haar Classifier - face)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ustawienia kamery (ENG. Camera settings)
cap = cv2.VideoCapture(0) # Wybór kamery (ENG. Camera selection - laptop camera)
cap.set(3,640) # Ustawienie szerokości (ENG. Width setting)
cap.set(4,480) # Ustawienie wysokości (ENG. Height setting)

# Dane potrzebne do wyznaczenia ilości FPS (ENG. Data needed to determine the amount of FPS )
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyświetlanych napisów (ENG. Setting the font of the displayed subtitles)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Uruchomienie kamery (ENG. Start the camera)
    _, img = cap.read()
    frame_id += 1
    
    # Konwersja do odcieni szrości (ENG. Convert to grayscale)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Definicja parametrów klasyfikatora Haara (ENG. Definition of Haar classifier parameters)
    faces = faceCascade.detectMultiScale(
        gray_img,        # Wejściowy obrazek w skali szarości (ENG. Grayscale input image)
        scaleFactor=1.2, # Parametr określający stopień zmniejszenia rozmiaru obrazu w każdej skali obrazu. Służy do tworzenia piramidy skali. (ENG. A parameter that specifies how much the image size is reduced for each image scale. Use to create a scale pyramid.)
        minNeighbors=5,  # Parametr określający, ilu sąsiadów powinien mieć każdy kandydujący prostokąt, aby go zachować (ENG. A parameter that specifies how many neighbors each candidate rectangle should have in order to keep it)
                         # Wyższa liczba daje mniej przypadków false positives (ENG. A higher number gives fewer cases of false positives) 
        minSize=(20,20),  # Minimalny rozmiar prostokąta, który należy uznać za twarz (ENG. The minimum size of the rectangle to be considered a face)
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Zaznaczenie za pomocą prostokąta twarzy (wykrywanie i zaznaczanie twarzy na obrazie) (ENG. Select face with rectangle (face detection and marking in image))
    for (x,y,w,h) in faces:
        
        # Wycięte zdjęcia twarzy (ENG. Cut out of the face photo)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
        
        facess = faceCascade.detectMultiScale(roi_gray)

        # Obróbka zdjęć twarzy (ENG. Processing face photos)
        final_image = cv2.resize(roi_color, (img_size,img_size))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image/255.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Predykcja - 0:'INCORRECT MASK', 1:'FACE MASK', 2:'NO MASK' (ENG. Prediction - 0: 'INCORRECT MASK', 1: 'FACE MASK', 2: 'NO MASK')
        predictions = model.predict(final_image)
        
        # Wybór największej wartości z tabeli i określenie, na którym miejscu się znajduje (ENG. Selecting the highest value in the table and determining where it is placed)
        label = np.argmax(predictions, axis=1)

        # Zmiana koloru prostokąta i wyświetlanie tekstu w zależności od predykcji (ENG. Change the color of the rectangle and display the text depending on the prediction)
        if label == 0:
            status = "INCORRECT MASK"
            # Narysowanie tła w kształcie prostokąta, na którym znajdować się będzie informacja o maseczce (ENG. Drawing a rectangular background with information about the mask)
            cv2.rectangle(img, (x-1,y+h), (x + w + 1, y + h + 50), (86,228,253), -1 )
            # Dodanie tekstu (ENG. Add text)
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
    
    # Liczba FPS (ENG. FPS number)
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4)         
            
    # Wyświetlanie wyników (ENG. Displaying the results)
    cv2.imshow('FACE MASK DETECTION',img)
    
    # Zakończenie pracy programu (ENG. Ending the program)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Zakończenie transmisji przez naciśnięcie klawisza 'ESC' (ENG. Ending the transmission by pressing the 'ESC' key)
        break
cap.release()
cv2.destroyAllWindows()


