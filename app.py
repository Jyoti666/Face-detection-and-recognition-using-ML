import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import cv2
import os
from os import listdir
from os.path import isfile, join
import face_recognition as fr
import face_recognition
from time import sleep








app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')





@app.route('/upload_img',methods=['POST'])
def upload_img():
    print(request.form.get('name'))
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_extractor(img):

        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #faces = face_classifier.detectMultiScale(gray,1.3,5)
        faces = face_classifier.detectMultiScale(img,1.3,5)
        if faces is():
            return None

        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face


    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            #file_name_path = './img/'+str(count)+'.jpg'
            file_name_path = './img/'+request.form.get('name')+'.jpg'

            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not found")
            pass
        #print(cv2.getWindowProperty('Face Cropper',cv2.WND_PROP_VISIBLE))#cv2.WND_PROP_AUTOSIZE))
        #print(cv2.getWindowProperty('Face Cropper', cv2.WND_PROP_AUTOSIZE))
        #if cv2.getWindowProperty('Face Cropper', cv2.WND_PROP_AUTOSIZE) < 1:
            #break
        prop_asp = cv2.getWindowProperty('Face Cropper', cv2.WND_PROP_ASPECT_RATIO)
        if prop_asp < 0.0:
            break
        if cv2.getWindowProperty('Face Cropper', 0)<0:
            break
        #if not cv2.getWindowProperty('Face Cropper', cv2.WND_PROP_VISIBLE):
            #break
        if cv2.waitKey(1)==13 or count==100:#(cv2.waitKey(1) & 0xFF == 27) or count==100:#cv2.waitKey(1) & 0xFF == ord('q'):
            #print("Key:",cv2.waitKey(1))
            #print()
            break
        #cv2.waitKey()
    cap.release()
    cv2.destroyAllWindows()
    print('Samples Colletion Completed ')
    #print(cv2.getWindowProperty('Face Cropper', cv2.WND_PROP_AUTOSIZE))
    return render_template('index.html')



@app.route('/video_recog',methods=['POST'])
def video_recog():\
    #print("--------------------")
    #print(request.form.get('name'))
    data_path = './img/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Dataset Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return img,[]

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))


#request.form.get('name')
            if confidence > 82:
                cv2.putText(image, (onlyfiles[result[0]].split("."))[0], (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1)==13:
            break


    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')







@app.route('/img_recog',methods=['POST'])
def img_recog():
    
    def get_encoded_faces():
        """
        looks through the faces folder and encodes all
        the faces

        :return: dict of (name, image encoded)
        """
        encoded = {}

        for dirpath, dnames, fnames in os.walk("./img"):
            for f in fnames:
                print("f=",f)
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file("img/" + f)
                    print("----------------------------")
                    #print(face)
#                   print(fr.face_encodings(face))
                    encoding = fr.face_encodings(face)[0]
                    print('1')
                    encoded[f.split(".")[0]] = encoding
                    print('2')
        return encoded


    def unknown_image_encoded(img):
        """
        encode a face given the file name
        """
        face = fr.load_image_file("img/" + img)
        encoding = fr.face_encodings(face)[0]

        return encoding


    def classify_face(im):
        """
        will find all of the faces in a given image and label
        them if it knows what they are

        :param im: str of file path
        :return: list of face names
        """
        faces = get_encoded_faces()
        faces_encoded = list(faces.values())
        known_face_names = list(faces.keys())
        print(known_face_names)
        img = cv2.imread(im.filename, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
        face_locations = face_recognition.face_locations(img)
        unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"

        # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
                cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
                cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
        while True:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', img)
            if cv2.waitKey(1)==13:#cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.waitKey()
                cv2.destroyAllWindows()
                break

    classify_face(request.files['img_recog'])
    #print("0000000000000000000000000000999999999999999999999999999999")
    return render_template('index.html')
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def a():    
    
    #import face_recognition as fr
    def get_encoded_faces():
    
        encoded = {}
        i=6
        for dirpath, dnames, fnames in os.walk("./img"):
            #print(fnames)
            j=22
            for f in fnames:
                #print(f)
                
                if f.endswith(".jpg") or f.endswith(".png"):
                    print("k")
                    face = fr.load_image_file("img/chintu.jpg")# + f)
                    #print(face)
                    a=fr.face_encodings(face)
                    print(a)
                    print("l")
                    encoding = fr.face_encodings(face)[0]
                    print("m")
                    encoded[f.split(".")[0]] = encoding
                    print("n")

        return encoded


    def unknown_image_encoded(img):
        """
        encode a face given the file name
        """
        face = fr.load_image_file("img/" + img)
        encoding = fr.face_encodings(face)[0]

        return encoding


    def classify_face(im):
        """
        will find all of the faces in a given image and label
        them if it knows what they are
        
        :param im: str of file path
        :return: list of face names
        """
        print(1)
        faces = get_encoded_faces()
        print(2)
        faces_encoded = list(faces.values())
        print(3)
        known_face_names = list(faces.keys())
        print(4)
        print(im.filename)
        img = cv2.imread(im.filename,1)#, 1)
        #cv2.imshow('image', img)
        print(5)
        print(img)
        face_locations = face_recognition.face_locations(img)
        print(6)
        unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
        print(7)
        face_names = []
        for face_encoding in unknown_face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"
            
            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            #face_names.append(request.form.get('name'))
            face_names.append(name)   
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
                    
                # Draw a label with a name below the face
                cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


        # Display the resulting image
        while True:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', img)
            if cv2.waitKey(1)==13:#cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.waitKey()
                cv2.destroyAllWindows()
                break 
            
    print("0000000000000000000000000000")   #f = request.files['image']  
    #f.save(f.filename)
    #cv2.imshow('image', str((request.files['img_recog']).filename))
    classify_face(request.files['img_recog'])#str(request.form.get('img_recog')))
    print("----------------------------------------------------------")
    return render_template('index.html')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
@app.route('/predict',methods=['POST'])
def predict():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print(1)
    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    # To use a video file as input
    print(2)
    while True:
    # Read the frame
        _, img = cap.read()
        print("=")
    # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print("*")
    # Display
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        print("&")
    # Stop if enter key is pressed
        #k = cv2.waitKey(30) & 0xff
        if cv2.waitKey(1)==13:
            break

    # Release the VideoCapture object
    cap.release()
    return render_template('index.html')    


@app.route('/img_detect',methods=['POST'])
def img_detect():
    
    f = request.files['image']  
    f.save(f.filename)
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read the input image
    img = cv2.imread(str(f.filename))

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the output
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey()
    return render_template('index.html')
    
    
if __name__ == "__main__":
    app.run()
