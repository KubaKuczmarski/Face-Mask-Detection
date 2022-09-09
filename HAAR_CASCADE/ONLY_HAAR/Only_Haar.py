# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:22:15 2022

Subject: Face mask detection using only Haar clasificators.

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek (ENG. Loading libraries)
import cv2
import time

# Wczytanie potrzebnych klasyfikatorów - usta, nos, twarz (ENG. Loading the necessary Haar classifiers - mouth, nose, face)
noseCascade = cv2.CascadeClassifier('nose.xml')
mouthCascade = cv2.CascadeClassifier('mouth.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Wybór kamery (ENG. Camera selection - laptop camera)
cap = cv2.VideoCapture(0)
cap.set(3,640) # Ustawienie szerokości (ENG. Width setting)
cap.set(4,480) # Ustawienie wysokości (ENG. Height setting)

# Dane potrzebne do wyznaczenia ilosci FPS (ENG. Data needed to determine the amount of FPS)
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyświetlanych napisów (ENG. Setting the font of the displayed subtitles)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Uruchomienie kamery (ENG. Start the camera)
    _, img = cap.read()
    frame_id += 1
    
    # Konwersja do odcieni szrości (ENG. Convert to grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Definicja parametrów klasyfikatora Haara - nos (ENG. Definition of Haar classifier parameters - nose)
    nose = noseCascade.detectMultiScale(
    gray, 
    scaleFactor=1.2,  
    minNeighbors=15,    
    minSize=(10, 10) 
    )
    
    # Definicja parametrów klasyfikatora Haara - usta (ENG. Definition of Haar classifier parameters - mouth)
    mouth = mouthCascade.detectMultiScale(
    gray, 
    scaleFactor=1.2,  
    minNeighbors=60,   
    minSize=(20, 20)
    )
    
    # Definicja parametrów klasyfikatora Haara - twarz (ENG. Definition of Haar classifier parameters - face)
    faces = faceCascade.detectMultiScale(
    gray,     
    scaleFactor=1.2,
    minNeighbors=5,    
    minSize=(20, 20)
    )
    
    sizeNose = 0
    for (x,y,w,h) in nose:    
        # Oznaczenie nosa na ekranie (ENG. Nose marking on the screen)
        noseRect = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # Definicja rozmiaru prostokąta (ENG. Rectangle size definition)
        sizeNose=noseRect.size 
        
          
    sizeMouth = 0
    for (x,y,w,h) in mouth:
        mouthRect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        sizeMouth=mouthRect.size 
     
    sizeFace = 0
    for (x,y,w,h) in faces:
        faceRect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        sizeFace = faceRect.size
        
    # Wybór odpowiedniej z opcji położenia maseczki na twarzy (ENG. Choosing the appropriate option for placing the mask on the face)
    if (sizeNose > 0 and sizeMouth == 0) or (sizeNose == 0 and sizeMouth > 0):
        status = "INCORRECT MASK"
        # Narysowanie tła w kształcie prostokąta, na ktorym znajdować się będzie informacja o maseczce (ENG. Drawing a rectangular background with information about the mask)
        cv2.rectangle(img, (x-1,y+h), (x + w + 1, y + h + 50), (86,228,253), -1 )
        # Dodanie tekstu (ENG. Add text)
        cv2.putText(img, status, (x+5, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255,255,255), 2)
        # Prostokąt oznaczający położenie twarzy (ENG. Rectangle representing the location of the face)
        cv2.rectangle(img, (x,y), (x+w, y+h), (86,228,253), 2)
    
    elif sizeNose > 0 and sizeMouth > 0:
         status = "NO MASK"
         cv2.rectangle(img, (x-1,y+h), (x + w + 1, y + h + 50 ), (0,0,255), -1 )
         cv2.putText(img, status, (x+50, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
         cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        
    elif sizeFace > 0 and sizeNose == 0 and sizeMouth == 0:
        status = "FACE MASK"
        cv2.rectangle(img, (x-1,y+h), (x + w + 1, y + h + 50 ), (0,255,0), -1 )
        cv2.putText(img, status, (x+50, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 1)
    
    # Liczba FPS (ENG. FPS number)
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4)
    
    # Wyświetlenie okna z widokiem z kamery i wynikami (ENG. Display a window with the camera view and results)
    cv2.imshow('FACE MASK DETECTION',img)
    
    # Zakończenie pracy programu (ENG. Ending the program)
    k = cv2.waitKey(30) & 0xff
    if k == 27: ## Zakończenie transmisji przez naciśnięcie klawisza 'ESC' (ENG. Ending the transmission by pressing the 'ESC' key)
        break
cap.release()
cv2.destroyAllWindows() 