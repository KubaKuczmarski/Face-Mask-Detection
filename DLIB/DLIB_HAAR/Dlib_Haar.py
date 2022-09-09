# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:54:07 2022

Subject: Face mask detection using dlib detector and Haar cascade clasificators (nose and mouth).

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek (ENG. Loading libraries)
import cv2
import numpy as np
import dlib
import time

# Wczytanie potrzebnych klasyfikatorów - usta, nos (ENG. Loading the necessary Haar classifiers - mouth, nose)
noseCascade = cv2.CascadeClassifier('nose.xml')
mouthCascade = cv2.CascadeClassifier('mouth.xml')

# Wybór kamery (ENG. Camera selection - laptop camera)
cap = cv2.VideoCapture(0)

# Dane potrzebne do wyznaczenia ilosci FPS (ENG. Data needed to determine the amount of FPS)
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyświetlanych napisów (ENG. Setting the font of the displayed subtitles)
font = cv2.FONT_HERSHEY_SIMPLEX

## Funkcja, która w zależności od opcji - NO MASK, FACE MASK lub INCORRECT MASK odpowiednio obrysowuje twarz i wyświetla informacje o sposobie założenia maseczki (ENG.A function that, depending on the options - NO MASK, FACE MASK or INCORRECT MASK, rightly contours the face and displays information about how the mask is wearing)
def switch(option ,_x1,_y1, _x2, _y2, result, image):
    
    w = _x2 - _x1 # Szerokość prostokata (ENG. Width of the rectangle)
    h = _y2 - _y1 # Wysokość prostokąta (ENG. The height of the rectangle)
    
    # Opcja INCORRECT MASK (ENG. INCORRECT MASK option)
    if option == 1:
        result = "INCORRECT MASK"
        # Narysowanie tła w kształcie prostokąta, na ktorym znajdować się będzie informacja o maseczce (ENG. Drawing a rectangular background with information about the mask)
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (86,228,253), -1 )
        # Dodanie tekstu (ENG. Add text)
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255,255,255), 2)
        # Prostokąt oznaczający położenie twarzy (ENG. Rectangle representing the location of the face)
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (86,228,253), 2)
    
    # Opcja NO MASK (ENG. NO MASK option)
    elif option == 2:
        result = "NO MASK"
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (0,0,255), -1 )
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (0,0,255), 2)

    # Opcja FACE MASK (ENG. FACE MASK option)
    elif option == 3:
        result = "FACE MASK"
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (0,255,0), -1 )
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (0,255,0), 2)

## Funkcja odpowiedzialna ze detekcję twarzy przy pomocy biblioteki dlib (ENG. Function responsible for face detection using the dlib library)
def face(_gray, _img, _color):
        
        # Detekcja twarzy (ENG. Face detection)
        faces = detector(_gray)
        
        # Określenie współrzednych twarzy (ENG. Determination of face coordinates)
        for face in faces:
            # Wyciągnięcie pojedyńczych punktów ze współrzędnych okręślających położenie twarzy (ENG. Take out single points from the coordinates defining the position of the face) 
            # Współrzędne (x,y) górnego lewego punktu (ENG. Coordinates (x, y) of the upper left point)
            x1 = face.left() 
            y1 = face.top()
            
            # Współrzędne (x,y) dolnego prawgo punktu (ENG. Coordinates (x, y) of the lower right point)
            x2 = face.right()
            y2 = face.bottom()
            
            switch(_color, x1, y1, x2, y2, " ", _img)
            
## Funkcja odpowiedzialna za detekcje ust - Haar Cascade (ENG. The function responsible for mouth detection - Haar Cascade)       
def mouth(_gray, _img):
    
    # Definicja parametrów klasyfikatora Haara służącego do wykrywania ust (ENG. Definition of Haar Classifier parameters for mouth detection)
    mouth = mouthCascade.detectMultiScale(
    _gray, 
    scaleFactor=1.2, 
    minNeighbors=60,  
    minSize=(20, 20), 
    flags=cv2.CASCADE_SCALE_IMAGE 
    ) 
    
    # Współrzędne prostokata, który obrysowywuje usta (ENG. The coordinates of the rectangle that outline the mouth) 
    mouthRect=np.array([[0, 0, 0], [0, 0, 0]]) # mouthRect - zmienna służąca określeniu, czy prostokat zaznacza usta na obrazku, czy nie (ENG. zmienna służąca określeniu, czy prostokat zaznacza usta na obrazku, czy nie)
        
    for (x,y,w,h) in mouth:
        
        # Wycięty fragment ust (ENG. A fragment of the mouth)
        mouth = _img [x:y, (x+w, y+h)]
        # Prostokąt określjący polożenie ust (ENG. Rectangle defining the position of the mouth)
        mouthRect = cv2.rectangle(_img,(x,y),(x+w,y+h),(0,255,0),2)
        mouthRect
    
    return mouthRect

## Funkcja odpowiedzialna za detekcje nos - Haar Cascade (ENG. Function responsible for nose detection - Haar Cascade)                 
def nose(_gray, _img):
    
    # Definicja parametrów klasyfikatora Haara służącego do wykrywania nosa (ENG. Definition of Haar Classifier parameters for nose detection)
    nose = noseCascade.detectMultiScale(
    _gray, # Wejściowy obrazek w skali szarości (ENG. Grayscale input image)
    scaleFactor=1.2,  
    minNeighbors=15,     
    minSize=(10, 10), 
    flags=cv2.CASCADE_SCALE_IMAGE
    )

    noseRect=np.array([[0, 0, 0], [0, 0, 0]])
    
    for (x,y,w,h) in nose:
        # Wycięty fragment nosa (ENG. A fragment of the nose)
        nose = _img [x:y, (x+w, y+h)]
        # Prostokąt określjący położenie nosa (ENG. Rectangle defining the position of the nose)
        noseRect = cv2.rectangle(_img,(x,y),(x+w,y+h),(255,0,0),2)
        noseRect
        
    return noseRect


######################## KONIEC DEKLARACJI FUNKCJI ######################################

# Wczytanie detektora
detector = dlib.get_frontal_face_detector()

while True:
    
    # Uruchomienie kamery (ENG. Start the camera)
    _, img = cap.read()
    frame_id += 1
    
    # Konwersja do odcieni szrości - łatwiejsza obróbka, mniejsze obciązenie CPU (ENG. Convert to grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Zmienna określająca, czy usta zostały zaznaczone na rysunku ( zaznzaczone - widac prostokat, mie ma maseczki = M.size = 92160, nie zaznaczone - nie ma prostokata, jest maseczka = M.size = 6 ) (ENG. Variable that determines whether the lips have been detected (detect - you can see the rectangle, no masks = M.size = 92160, undetected - no rectangle, there is a mask = M.size = 6))
    M = mouth(gray, img) 
    N=nose(gray, img)
     
    # Warunek INCORRECT MASK - odpowiednia funkcja face zaznaczająca twarz i wyświetlajaca informacje (ENG. INCORRECT MASK condition)
    if ((M.size <= 6) and (N.size > 6)) or ((M.size > 6) and (N.size <= 6)):
        face(gray, img, 1)
    # Warunek NO MASK (ENG. NO MASK condition)
    elif (M.size > 6) and (N.size > 6):
       face(gray, img, 2)
    #Warunek FACE MASK (ENG. FACE MASK condition)
    elif (M.size <= 6) and (N.size <= 6):
        face(gray, img, 3)
    
    # Liczba FPS (ENG. FPS number)
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4) 
    
    # Wyświetlenie okna z widokiem z kamery i wynikami (ENG. Display a window with the camera view and results)
    cv2.imshow("FACE MASK DETECTION", img)
    
    # Zakńoczenie pracy programu (ENG. Ending the program)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Zakończenie transmisji przez naciśnięcie klawisza 'ESC' (ENG. Ending the transmission by pressing the 'ESC' key)
        break
cap.release()
cv2.destroyAllWindows()