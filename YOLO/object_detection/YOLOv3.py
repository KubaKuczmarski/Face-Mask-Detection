# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:14:13 2022

Subject: Face mask detection using YOLO algorithm.

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek (ENG. Loading libraries)
import cv2
import numpy as np
import dlib
import time

# Wczytanie wag sieci i pliku konfiguracyjnego YOLOv3 (ENg. Uploading network scales and YOLOv3 configuration file)
net = cv2.dnn.readNet("D:\studia\INZYNIERKA\PRACA\PRACA_INZYNIERSKA\YOLO\YOLO\yolov3\weights\yolov3_custom_train_final.weights", "D:\studia\INZYNIERKA\PRACA\PRACA_INZYNIERSKA\YOLO\YOLO\yolov3\cfg\yolov3_custom_test.cfg")

# Deklaracja niestandardowych klas dla wytrenowanego modelu (ENG. Declaration of custom classes for a trained model)
classes = ['MASK','NOSE','MOUTH']

# Wybór kamery (ENG. Camera selection - laptop camera)
cap = cv2.VideoCapture(0)

# Dane potrzebne do wyznaczenia ilości FPS (ENG. Data needed to determine the amount of FPS )
starting_time = time.time() 
frame_id = 0 

# Wczytanie detektora dlib (ENG. Loading the dlib detector)
detector = dlib.get_frontal_face_detector()

# Ustawienie czcionki wyświetlanych napisów (ENG. Setting the font of the displayed subtitles)
font = cv2.FONT_HERSHEY_SIMPLEX

# Ustawienie rozmiaru zdjęcia (ENG. Set the size of the photo)
img_size = 150

# Parametr pomocniczy (ENG. Auxiliary parameter)
param = 'Non'

# Lista zawierajaca dwa kolejne obiekty wykryte przez program. Wykorzystana podczas predykcji wyników. (ENG. A list containing two consecutive objects detected by the program. Used in predicting results.)
object = ['Non', 'Non']
    
while True:
    # Uruchomienie kamery (ENG. Start the camera)
    _, img = cap.read() 
    frame_id += 1
    
    ### DLIB
    
    # Konwersja do odcieni szrości (ENG. Convert to grayscale)
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
    
    ### YOLO
    
    # Wysokość, szerokość i liczba kanałów (RGB) pobranego obrazka z kamery (ENG. Height, width and number of channels (RGB) of the downloaded image from the camera)
    height, width, channels = img.shape
    
    # Wykrywanie obiektów (ENG. Object detection)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    
    # Wyjście z pojedyńczej warstwy (ENG. Exit from a single layer)
    outs = net.forward(output_layers)
    
    # Tablice przechowywujące konkretne wartości (ENg. Arrays that store specific values)
    class_ids = [] # Rodzaj klasy 0: MASK, 1: NOSE, 2: MOUTH (ENG. Class type 0: MASK, 1: NOSE, 2: MOUTH)
    confidences = [] # Wartość określająca pewność przynależności do danej klasy wykrytego obiektu (ENG. Value specifying the certainty of belonging to a given class of the detected object)
    boxes = [] # Współrzędne i wartości prostokąta określajacego połorzenie danego obiektu (ENG. Coordinates and values of a rectangle defining the location of a given object)
    for out in outs:
        for detection in out:
            scores = detection[5:] 
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:  
                
                # Współrzędne środka wykrytego obiektu (ENG. Coordinates of the center of the detected object)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                # Szerokość prostokąta (ENG. The width of the rectangle)
                w = int(detection[2] * width)
                
                # Wysokość prostokąta (ENG. The height of the rectangle)
                h = int(detection[3] * height)
                
                # Współrzędne początkowe prostokąta (górny lewy róg) (ENG. Rectangle initial coordinates (upper left corner))
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Dodajemy poszczególne wartości do określonej tablicy (ENG. We add individual values to a specific array)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Usuwanie szumów - duplikujących się prostokątów z obrazka (ENG. Noise removal - duplicating rectangles in an image)  
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    ### DLIB - wyświetlanie wyników (ENG. DLIB - results display)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            param = object[0]
            object[0]=label
            object[1]=param
            confidence = str(round(confidences[i],2))
            
            # Wybór jednej z opcji połozenia maseczki na twarzy (ENG. Choosing one of the options for placing the mask on the face)
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
    
    # Liczba FPS (ENG. FPS number)
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4)
    
    cv2.imshow("FACE MASK DETECTION", img)
    key = cv2.waitKey(1)
    if key == 27: # ASCI = 27 -> 'ESC'
        break;
        
cap.release()
cv2.destroyAllWindows()
