# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:06:48 2022

Subject: Face mask detection using Face Landmarks Detection.

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek
import cv2
import dlib
import matplotlib.pyplot as plt
from PIL import Image
import time

# Wybór kamery
cap = cv2.VideoCapture(0)

# Dane potrzebne do wyznaczenia ilosci FPS
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyswietlanych napisów
font = cv2.FONT_HERSHEY_SIMPLEX

## Funkcja, która w zaleznoci od opcji - NO MASK, FACE MASK lub INCORRECT MASK odpowiednio obrysowuje twarz i wyswietla informacje o soposobie zalozenia maseczki
def switch(option ,_x1,_y1, _x2, _y2, image):
    
    w = _x2 - _x1 #szerokosc prostokata
    h = _y2 - _y1 #wysokosc prostokata
    
    # Opcja INCORRECT MASK
    if option == 1:
        result = "INCORRECT MASK"
        #Obramowanie twarzy
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (86,228,253), 2)
        #Zamalowany prostokąt na tekst
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (86,228,253), -1 )
        #Tekst
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255,255,255), 2)
 
    elif option == 2:
        result = "NO MASK"
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (0, 0, 255), 2)
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (0,0,255), -1 )
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 
 
    elif option == 3:
        result = "FACE MASK"
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (0,255,0), 2)
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (0,255,0), -1 )
        cv2.putText(image, result, (_x1+5, _y1 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

## Funkcja wyswietlajaca instrukcje skierowana do osoby sprawdzajacej czy ma maseczke n twarzy
def instruction():
    # Załadowanie obrazka z instrukcja
    inst = cv2.imread("./instruction.jpg")

    # Wyswietlenie instrukcji
    cv2.imshow("INSTRUKCJA", inst)
    cv2.waitKey(0)

# Komponenty biblioteki dlib
detector = dlib.get_frontal_face_detector() # wczytanie detektora

# Wczytanie metody Face Landmarks Detection
shape_predictor = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor)

# Funckcja sprawdzajaca pojedyncze wartosci RGB punktow Landmarks - usta, nos
def colorPixel(img_path, x, y, _string, _partOfFace):
    img = Image.open(img_path).convert('RGB') # Konwersja obrazka do RGB
    r, g, b = img.getpixel((x, y))
    RGB = (r, g, b) 
    return RGB, _string, _partOfFace #Zwraca wartosci RGB w postaci tupla i nazwe czesci ciala, ktorej dotyczy sprawdzany punkt - nos, usta

## Funcka nanoszaca punkty Landmarks i odwolujaca sie do funkcji colorPixel w celu sprawdzenia wartosci RGB
def colorDetection (_img, _name, _nose, _mouth):
    ready = True
    while ready:
        img = _img
        frame = cv2.imread(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #hsv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            
            # 7, 8, 9 - broda
            # 30 , 31, 32, 33, 34, 35 - nos
            # 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67 - usta
            # [30 , 31, 32, 33, 34, 35, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
            
            pikselTable = [] # Pusta tabela na wartosci RGB, nazwe zdjecia i czesci ciala - np. [((232, 151, 134), 'Face', 'Nos'), ((167, 75, 78), 'Face', 'Usta')]
            
            pixelNumber = [30, 62] # 30 - nos, 62 - usta
            partOfFace = "None"
            for n in pixelNumber:
                if n == 30:
                    partOfFace = _nose # Dla n=30 nazwa - 'Nos'
                else:
                    partOfFace = _mouth
                # Wspolrzedne x, y punktu 
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) # Zaznaczenie punktu na obrazku
                
                pikselTable.append(colorPixel(img, x, y, _name, partOfFace)) # dodanie do tabeli wartosci zwroconych przez funkcje colorPixel
                
        # Wyswietlnir zdjecia w terminalu z naniesionymi punktami
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()
        ready = False
        
    return pikselTable # Zwrocenie calej tabeli z wartosciami pikseli
    
instruction() # Wywolanie funkcji instrukcja

ready = True
while ready:
    # Uruchomienie kamery
    _, frame = cap.read()
    frame_id += 1
    
    # Konwersja do odcieni szrosci 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # Detekcja twarzy
    faces = detector(gray)
    for face in faces:
        # Wyciągamy pojedyńcze punkty ze wspołrzędnych okręlajacych połozenie twarzy 
        # Wspołrzędne (x,y) górnego lewego punktu
        x1 = face.left() 
        y1 = face.top()
        # Wspołrzedne (x,y) dolnego prawgo punktu
        x2 = face.right()
        y2 = face.bottom()
        
        fileName="face.jpg" # Nazwa zapisanego zdjecia do sprawdzenia
        
        print("[INFO] wykrywanie twarzy ...") # Informacja zwrotna co sie dzieje w programie
        
        rectangle = frame[y1-98:y2+29, x1-28:x2+19] # Prostokat obejmujacy twarz do wyciecia i zapisaniea do dalszej obróbki
        cv2.rectangle(frame, (x1-30,y1-100), (x2+20,y2+30), (0, 0, 0), 2) # Zaznaczenie twarzy prostokatem
        
        print("[INFO] Zapis zdjęcia twarzy ...")
        
        #Zapisywanie zdjęcia w galerii
        cv2.imwrite(fileName, rectangle)
        
        # Odwolanie do funkcji colorDetection - zwraca [((232, 151, 134), 'Face', 'Nos'), ((167, 75, 78), 'Face', 'Usta')]
        face = colorDetection(fileName, "Face", "Nos", "Usta")
        print(face) 
        
        # Podzial wartosci na pojedyncze wartosci RGB
        NoseR = face[0][0][0] #232
        NoseG = face[0][0][1] #151
        NoseB = face[0][0][2] #134
            
        MouthR = face[1][0][0] #167
        MouthG = face[1][0][1] #75
        MouthB = face[1][0][2] #78

        # Warunek NO MASK
        if 196<NoseR<205 and 100<NoseG<110 and 70<NoseB<78 and 134<MouthR<163 and 29<MouthG<45 and 32<MouthB<47:
            switch(2,x1-30,y1-100, x2+20, y2+30, frame)
        # Warunek INCORRECT MASK - odpowiednia funkcja face zaznaczajaca twarz i wyswietlajaca informacje
        elif 200<NoseR<231 and 87<NoseG<143 and 77<NoseB<105 and 207<MouthR<235 and 144<MouthG<216 and 128<MouthB<149:
            switch(1,x1-30,y1-100, x2+20, y2+30, frame)
        # Waunek FACE MASK
        elif 220<NoseR<245 and 173<NoseG<190 and 135<NoseB<166 and 156<MouthR<222 and 112<MouthG<173 and 113<MouthB<125:
            switch(3,x1-30,y1-100, x2+20, y2+30, frame)   
            
    # Liczba FPS
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4) 

    # Wyswietlenie okna z widokiem z kamery i wynikami
    cv2.imshow("FACE MASK DETECTION", frame)
    
    # Zakonczenie pracy programu
    k = cv2.waitKey(30) & 0xff
    if k == 27: # zakończenie transmidji przez nacinięcie klawisza 'ESC'
        break
cap.release()
cv2.destroyAllWindows()

