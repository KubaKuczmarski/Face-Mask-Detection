# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:14:13 2022

Subject: Face mask detection using YOLO algorithm.

@author: Kuba Kuczmarski
"""

# Wcytanie bibliotek
import cv2
import numpy as np
import dlib
import time

# Wczytanie sieci wag i pliku konfiguracyjnego YOLOv3
net = cv2.dnn.readNet("D:\studia\INZYNIERKA\PRACA\PRACA_INZYNIERSKA\YOLO\YOLO\yolov3\weights\yolov3_custom_train_final.weights", "D:\studia\INZYNIERKA\PRACA\PRACA_INZYNIERSKA\YOLO\YOLO\yolov3\cfg\yolov3_custom_test.cfg")

# Deklaracja niestandardowych klas dla wytrenowanego modelu
classes = ['MASK','NOSE','MOUTH']

# Uruchomienie kamery
cap = cv2.VideoCapture(0)

# Dane potrzebne do wyznaczenia ilosci FPS
starting_time = time.time() 
frame_id = 0 

# Wczytanie detektora dlib
detector = dlib.get_frontal_face_detector()

# Ustawienie czcionki
font = cv2.FONT_HERSHEY_SIMPLEX

# Rozmiar zdjęcia
img_size = 150

# Parametr pomocniczy 
param = 'Non'
# Lista zawierajaca dwa kolejne obiekty wtkryte przez program. Wykorzytana podczas predykcji wyników. 
object = ['Non', 'Non']
    
while True:
    _, img = cap.read() 
    frame_id += 1
    
    ### DLIB
    
    # Konwersja do odcieni szrosci 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
    
    ### YOLO
    
    # Wysokoc, szerokoc i liczba kanałów (RGB) pobranego obrazka z kamery
    height, width, channels = img.shape
    
    # wykrywanie obiektów 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    
    # Wyjcie z pojedyńczej warstwy
    outs = net.forward(output_layers)
    
    # Tablice przechowywujące konkretne wartosci
    class_ids = [] # Rodzaj klasy 0: MASK, 1: NOSE, 2: MOUTCH
    confidences = [] # Wartosć okreslajaca pewnosc przynaleznosci do danej klasy wykrytego obiektu
    boxes = [] # Współrzedne i wartosci prostokąta okreslajacego połorzenie danego obiektu
    for out in outs:
        for detection in out:
            scores = detection[5:] 
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:  # 0.5 - treshold - mniejsza wartosc = wiecej obiektow, ale mniejsza dokladnosc, wieksza wartosc = mniej obiektow,ale wieksza dokladnosc
                
                # Współrzędne srodka wykrytego obiektu
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                # Szerokosć prostokąta
                w = int(detection[2] * width)
                
                # Wysokosć prostokąta
                h = int(detection[3] * height)
                
                # Współrzędne początkowe prostokąta (górny lewy róg)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Dodajemy poszczególne wartoci do okrelonwj tablicy 
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Usuwanie szumów - duplikujących się prostokątów z obrazka     
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    ### DLIB - wywietlanie wyników
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            param = object[0]
            object[0]=label
            object[1]=param
            confidence = str(round(confidences[i],2))
            
            # Wybór jednej z opcji połozeniamaseczki na twarzy
            if (object[0] == 'NOSE' and object[1] == "Non") or (object[0] == 'NOSE' and object[1] == "MASK") or (object[0] == 'MASK' and object[1] == "NOSE") or (object[0] == 'NOSE' and object[1] == "NOSE") or (object[0] == 'MOUTH' and object[1] == "MASK") or  (object[0] == 'MASK' and object[1] == "MOUTH"):
                cv2.rectangle(img, (x1,y1), (x2,y2), (86,228,253), 2)
                cv2.rectangle(img, (x1-1,y2), (x2+1, y2 + 50), (86,228,253), -1 )
                cv2.putText(img, "INCORRECT MASK", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                
            elif (object[0] == 'MASK' and object[1] == "Non") or (object[0] == 'MASK' and object[1] == "MASK"):
                cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1-1,y2), (x2+1, y2 + 50), (0,255,0), -1 )
                cv2.putText(img, "FACE MASKA", (x1+5, y2+30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            elif (object[0] == 'MOUTH' and object[1] == "Non") or (object[0] == 'NOSE' and object[1] == "MOUTH") or (object[0] == 'MOUTH' and object[1] == "NOSE") or (object[0] == 'Non' and object[1] == "Non") or (object[0] == 'MOUTH' and object[1] == "MOUTH") or (object[0] == 'MOUTH' and object[1] == "MASK") or (object[0] == 'MASK' and object[1] == "MOUTH"):
                cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1-1,y2), (x2+1, y2 + 50), (0,0,255), -1 )
                cv2.putText(img, "NO MASK", (x1+5, y2+30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    # Liczba FPS
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4)
    
    cv2.imshow("FACE MASK DETECTION", img)
    key = cv2.waitKey(1)
    if key == 27: # ASCI = 27 -> 'ESC'
        break;
        
cap.release()
cv2.destroyAllWindows()
