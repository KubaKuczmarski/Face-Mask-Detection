# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:54:07 2022

Subject: Face mask detection using dlib detector and Haar cascade clasificators (nose and mouth).

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek
import cv2
import numpy as np
import dlib
import time

# Wczytanie potrzebnych klasyfikatorów - usta, nos
noseCascade = cv2.CascadeClassifier('nose.xml')
mouthCascade = cv2.CascadeClassifier('mouth.xml')

# Wybór kamery
cap = cv2.VideoCapture(0)

# Dane potrzebne do wyznaczenia ilosci FPS
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyswietlanych napisów
font = cv2.FONT_HERSHEY_SIMPLEX

## Funkcja, która w zaleznoci od opcji - NO MASK, FACE MASK lub INCORRECT MASK odpowiednio obrysowuje twarz i wyswietla informacje o soposobie zalozenia maseczki
def switch(option ,_x1,_y1, _x2, _y2, result, image):
    
    w = _x2 - _x1 #szerokosć prostokata 
    h = _y2 - _y1 #wysokosc prostokąta 
    
    # Opcja INCORRECT MASK
    if option == 1:
        result = "INCORRECT MASK"
        # Prostokąt oznaczający położenie twarzy
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (86,228,253), 2)
        # Narysowanie tła w kształcie prostokąta, na ktorym znajdować się będzie informacja o maseczce
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (86,228,253), -1 )
        # Dodanie tekstu
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255,255,255), 2)
 
    elif option == 2:
        result = "NO MASK"
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (0,0,255), 2)
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (0,0,255), -1 )
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 
    elif option == 3:
        result = "FACE MASK"
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (0,255,0), 2)
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (0,255,0), -1 )
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 

## Funkcja odpowiedzialna ze detekcję twarzy przy pomocy biblioteki dlib
def face(_gray, _img, _color):
        
        # Detekcja twarzy
        faces = detector(_gray)
        
        # Okreslenie wspolrzednych twarzy
        for face in faces:
            # wyciągamy pojedyńcze punkty ze wspołrzędnych okręlajacych połozenie twarzy 
            # wspołrzędne (x,y) górnego lewego punktu
            x1 = face.left() 
            y1 = face.top()
            
            # wspołrzedne (x,y) dolnego prawgo punktu
            x2 = face.right()
            y2 = face.bottom()
            
            switch(_color, x1, y1, x2, y2, " ", _img)
            
## Funkcja odpowiedzialna za detekcje ust - Haar Cascade        
def mouth(_gray, _img):
    
    # Definicja parametrów klasyfikatora Haara służącego do wykrywania ust
    mouth = mouthCascade.detectMultiScale(
    _gray, 
    scaleFactor=1.2, #is the parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid. #1.2
    minNeighbors=60,  #is a parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives. #60  
    minSize=(20, 20), #is the minimum rectangle size to be considered a face. # 20,20
    flags=cv2.CASCADE_SCALE_IMAGE 
    ) 
    
    # Współrzędne prostokata, który obrysowuje usta
    mouthRect=np.array([[0, 0, 0], [0, 0, 0]]) # mouthRect - zmienna służąca okresleniu czy prostokat zaznacza usta na obrazku, czy nie 
        
    for (x,y,w,h) in mouth:
        
        # Wycięty fragment ust
        mouth = _img [x:y, (x+w, y+h)]
        # Prostkąt okresljący polożenie ust
        mouthRect = cv2.rectangle(_img,(x,y),(x+w,y+h),(0,255,0),2)
        mouthRect
    
    return mouthRect

## Funkcja odpowiedzialna za detekcje nos - Haar Cascade                  
def nose(_gray, _img):
    
    # Definicja parametrów klasyfikatora Haara służącego do wykrywania nosa
    nose = noseCascade.detectMultiScale(
    _gray, 
    scaleFactor=1.2,  
    minNeighbors=15,    
    minSize=(10, 10), 
    flags=cv2.CASCADE_SCALE_IMAGE
    )

    noseRect=np.array([[0, 0, 0], [0, 0, 0]])
    
    for (x,y,w,h) in nose:
        # Wycięty fragment nosa
        nose = _img [x:y, (x+w, y+h)]
        # Prostkąt okresljący polożenie nosa
        noseRect = cv2.rectangle(_img,(x,y),(x+w,y+h),(255,0,0),2)
        noseRect
        
    return noseRect


######################## KONIEC DEKLARACJI FUNKCJI ######################################

# Wczytanie detektora
detector = dlib.get_frontal_face_detector()

while True:
    
    # Uruchomienie kamery
    _, img = cap.read()
    frame_id += 1
    
    # Konwersja do odcieni szrosci - łatwiejsza obróbka, mniejsze obciązenie CPU
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Zmiena okreslająca, czy usta zostalły zaznaczone na rysunku ( zanzaczone - widac prostokat, mie ma maseczki = M.size = 92160, nie zaznaczone - nie ma prostokata, jest maseczka = M.size = 6 )
    M = mouth(gray, img) 
    N=nose(gray, img)
     
    # Warunek INCORRECT MASK - odpowiednia funkcja face zaznaczajaca twarz i wyswietlajaca informacje
    if ((M.size <= 6) and (N.size > 6)) or ((M.size > 6) and (N.size <= 6)):
        face(gray, img, 1)
    # Warunek NO MASK
    elif (M.size > 6) and (N.size > 6):
       face(gray, img, 2)
    #Warunek FACE MASK
    elif (M.size <= 6) and (N.size <= 6):
        face(gray, img, 3)
    
    # Liczba FPS
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4) 
    
    #Wywietlenie okna z widokiem z kamery i wynikami
    cv2.imshow("FACE MASK DETECTION", img)
    
    # Zakonczenie pracy programu
    k = cv2.waitKey(30) & 0xff
    if k == 27: # zakończenie transmidji przez nacinięcie klawisza 'ESC'
        break
cap.release()
cv2.destroyAllWindows()