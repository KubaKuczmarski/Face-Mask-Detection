# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:06:48 2022

Subject: Face mask detection using Face Landmarks Detection.

@author: Kuba Kuczmarski
"""

# Wczytanie bibliotek (ENG. Loading libraries)
import cv2
import dlib
import matplotlib.pyplot as plt
from PIL import Image
import time

# Wybór kamery (ENG. Camera selection - laptop camera)
cap = cv2.VideoCapture(0)

# Dane potrzebne do wyznaczenia ilości FPS (ENG. Data needed to determine the amount of FPS )
starting_time = time.time() 
frame_id = 0 

# Ustawienie czcionki wyświetlanych napisów (ENG. Setting the font of the displayed subtitles)
font = cv2.FONT_HERSHEY_SIMPLEX

## Funkcja, która w zależności od opcji - NO MASK, FACE MASK lub INCORRECT MASK odpowiednio obrysowuje twarz i wyświetla informacje o sposobie założenia maseczki (ENG. Setting the font of the displayed subtitles)
def switch(option ,_x1,_y1, _x2, _y2, image):
    
    w = _x2 - _x1 # Szerokość prostokata (ENG. Width of the rectangle)
    h = _y2 - _y1 # Wysokość prostokata (ENG. The height of the rectangle)
    
    # Opcja INCORRECT MASK (ENG. INCORRECT MASK option)
    if option == 1:
        result = "INCORRECT MASK"
        # Obramowanie twarzy (ENG. Face framing)
        cv2.rectangle(image, (_x1,_y1), (_x2,_y2), (86,228,253), 2)
        # Narysowanie tła w kształcie prostokąta, na ktorym znajdować się będzie informacja o maseczce (ENG. Drawing a rectangular background with information about the mask)
        cv2.rectangle(image, (_x1-1,_y1+h), (_x1 + w + 1, _y1 + h + 50 ), (86,228,253), -1 )
        # Dodanie tekstu (ENG. Add text)
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

## Funkcja wyświetlająca instrukcje skierowaną do osoby poddajacej się kontroli położenia maseczki na twarzy (ENG. A function that displays instructions to the person who controls the right position mask on the face)
def instruction():
    # Załadowanie obrazka z instrukcją (ENG. Uploading the picture with the instruction)
    inst = cv2.imread("./instruction.jpg")

    # Wyświetlanie instrukcji (ENG. Show the instruction)
    cv2.imshow("INSTRUKCJA", inst)
    cv2.waitKey(0)

# Komponenty biblioteki dlib (ENG. Dlib library components)
detector = dlib.get_frontal_face_detector() # Wczytanie detektora (ENG. Loading the detector)

# Wczytanie metody Face Landmarks Detection (ENG. Loading the Face Landmarks Detection method)
shape_predictor = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor)

# Funkcja sprawdzająca pojedyńcze wartości RGB punktów Landmarks - usta, nos (ENG. A function that checks individual RGB values of Landmarks points - mouth, nose)
def colorPixel(img_path, x, y, _string, _partOfFace):
    img = Image.open(img_path).convert('RGB') # Konwersja obrazka do RGB (ENG. Convert an image to RGB)
    r, g, b = img.getpixel((x, y))
    RGB = (r, g, b) 
    return RGB, _string, _partOfFace # Zwraca wartości RGB w postaci tupla i nazwe cześci ciała, której dotyczy sprawdzany punkt - nos, usta (ENG. Returns RGB values in the form of a tuple and the name of the body part affected by the point being checked - nose, mouth)

## Funkcja nanosząca punkty Landmarks i odwołujaca sie do funkcji colorPixel w celu sprawdzenia wartości RGB (ENG. A function that places landmarks and refers to the colorPixel function to check RGB values)
def colorDetection (_img, _name, _nose, _mouth):
    ready = True
    while ready:
        img = _img
        frame = cv2.imread(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            pikselTable = [] # Pusta tabela na wartości RGB, nazwę zdjęcia i części ciała - np. [((232, 151, 134), 'Face', 'Nos'), ((167, 75, 78), 'Face', 'Usta')] (ENG. Empty table for RGB values, photo name and body part - e.g. [((232, 151, 134), 'Face', 'Nose'), ((167, 75, 78), 'Face', 'Mouth') ])
            pixelNumber = [30, 62] # 30 - nos, 62 - usta (ENG. 30 - nose, 62 - mouth)
            partOfFace = "None"
            for n in pixelNumber:
                if n == 30:
                    partOfFace = _nose 
                else:
                    partOfFace = _mouth
                # Współrzędne x, y punktu (ENG. The x, y coordinates of the point)
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) # Zaznaczenie punktu na obrazku (ENG. Marking a point in the image)
                
                pikselTable.append(colorPixel(img, x, y, _name, partOfFace)) # Dodanie do tabeli wartości zwracanych przez funkcje colorPixel (ENG. Adding the values returned by the color Pixel functions to the table)
                
        # Wyświetlanie zdjęcia w terminalu z naniesionymi punktami (ENG. Displaying the picture in the terminal with the dots marked on it)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()
        ready = False
        
    return pikselTable # Zwrócenie całej tabeli z wartościami pikseli (ENG. Return the whole table with pixel values)
    
instruction() # Wywołanie funkcji - instrukcja (ENG. Function instruction)

ready = True
while ready:
    # Uruchomienie kamery (ENG. Start the camera)
    _, frame = cap.read()
    frame_id += 1
    
    # Konwersja do odcieni szrości (ENG. Convert to grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # Detekcja twarzy (ENG. Face detection)
    faces = detector(gray)
    for face in faces:
        # Wyciągamy pojedyńcze punkty ze współrzędnych okręślajacych położenie twarzy (ENG. We take out single points from the coordinates defining the position of the face)
        
        # Współrzędne (x,y) górnego lewego punktu (ENG. Coordinates (x, y) of the upper left point)
        x1 = face.left() 
        y1 = face.top()
        
        # Współrzedne (x,y) dolnego prawego punktu (ENG. Coordinates (x, y) of the lower right point)
        x2 = face.right()
        y2 = face.bottom()
        
        fileName="face.jpg" # Nazwa zapisanego zdjęcia do sprawdzenia (ENG. The name of the saved picture to be checked)
        
        print("[INFO] wykrywanie twarzy ...") # Informacja zwrotna co sie dzieje w programie (ENG. Feedback on what is happening in the program)
        
        rectangle = frame[y1-98:y2+29, x1-28:x2+19] # Zapisanie wycietego , prostokatnego fragmentu twarzy do dalszej obróbki (ENG. Save the cut, rectangular part of the face for further processing)
        cv2.rectangle(frame, (x1-30,y1-100), (x2+20,y2+30), (0, 0, 0), 2) # Zaznaczenie twarzy prostokatem (ENG. Mark the face with a rectangle)
        
        print("[INFO] Zapis zdjęcia twarzy ...")
        
        # Zapisywanie zdjęcia (ENG. Save a photo)
        cv2.imwrite(fileName, rectangle)
        
        # Odwołanie do funkcji colorDetection - zwraca [((232, 151, 134), 'Face', 'Nos'), ((167, 75, 78), 'Face', 'Usta')] (ENG. ColorDetection function reference - returns [((232, 151, 134), 'Face', 'Nose'), ((167, 75, 78), 'Face', 'Mouth')])
        face = colorDetection(fileName, "Face", "Nos", "Usta")
        print(face) 
        
        # Podział wartości na pojedyńcze wartości RGB (ENG. Division of values into individual RGB values)
        NoseR = face[0][0][0] #232
        NoseG = face[0][0][1] #151
        NoseB = face[0][0][2] #134
            
        MouthR = face[1][0][0] #167
        MouthG = face[1][0][1] #75
        MouthB = face[1][0][2] #78

        # Warunek NO MASK (ENG. NO MASK condition)
        if 196<NoseR<205 and 100<NoseG<110 and 70<NoseB<78 and 134<MouthR<163 and 29<MouthG<45 and 32<MouthB<47:
            switch(2,x1-30,y1-100, x2+20, y2+30, frame)
        # Warunek INCORRECT MASK (ENG. INCORRECT MASK condition)
        elif 200<NoseR<231 and 87<NoseG<143 and 77<NoseB<105 and 207<MouthR<235 and 144<MouthG<216 and 128<MouthB<149:
            switch(1,x1-30,y1-100, x2+20, y2+30, frame)
        # Warunek FACE MASK (ENG. FACE MASK condition)
        elif 220<NoseR<245 and 173<NoseG<190 and 135<NoseB<166 and 156<MouthR<222 and 112<MouthG<173 and 113<MouthB<125:
            switch(3,x1-30,y1-100, x2+20, y2+30, frame)   
            
    # Liczba FPS (ENG. FPS number)
    elapsed_time = time.time() - starting_time 
    fps = frame_id/elapsed_time 
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10,450), font, 1, (0,0,0), 4) 

    # Wyświetlenie okna z widokiem z kamery i wynikami (ENG. Display a window with the camera view and results)
    cv2.imshow("FACE MASK DETECTION", frame)
    
    # Zakończenie pracy programu (ENG. Ending the program)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Zakończenie transmisji przez naciśnięcie klawisza 'ESC' (ENG. Ending the transmission by pressing the 'ESC' key)
        break
cap.release()
cv2.destroyAllWindows()

