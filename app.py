from flask import Flask, render_template, Response, request, jsonify,redirect
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

app = Flask(__name__)
model = load_model('ML/ASL.h5')
cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        h, w ,c= frame.shape  
        # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html',prediction_text='')

@app.route('/videofeed')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    success, frame = cap.read()
    if not success:
        return jsonify({"character": "Error", "confidence": 0})
    h, w ,c= frame.shape  

    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_counter = 0
    analysisframe = ''
    letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    print("Space hit, capturing...")    
    analysisframe = frame
    framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
    resultanalysis = hands.process(framergbanalysis)
    hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
    if hand_landmarksanalysis:
        for handLMsanalysis in hand_landmarksanalysis:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lmanalysis in handLMsanalysis.landmark:
                x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20 

    analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
    analysisframe = analysisframe[y_min:y_max, x_min:x_max]
    analysisframe = cv2.resize(analysisframe,(64,64))
    nlist = []
    rows,cols = analysisframe.shape
    for i in range(rows):
        for j in range(cols):
            k = analysisframe[i,j]
            nlist.append(k)
        
    datan = pd.DataFrame(nlist).T
    colname = []
    for val in range(4096):
        colname.append(val)
    datan.columns = colname

    pixeldata = datan.values
    pixeldata = pixeldata/ 255
    pixeldata = pixeldata.reshape(-1,64,64,1)
    prediction = model.predict(pixeldata)
    predarray = np.array(prediction[0])
    letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
    predarrayordered = sorted(predarray, reverse=True)
    high1 = predarrayordered[0]
    high2 = predarrayordered[1]
    high3 = predarrayordered[2]
    for key,value in letter_prediction_dict.items():
        if value==high1:
            print("Predicted Character 1: ", key)
            print('Confidence 1: ', 100*value)
            character = key
            confidence = 100 * value
            return jsonify({"character": character, "confidence": confidence})
        elif value==high2:
            print("Predicted Character 2: ", key)
            print('Confidence 2: ', 100*value)
            character = key
            confidence = 100 * value
            return jsonify({"character": character, "confidence": confidence})
        elif value==high3:
            print("Predicted Character 3: ", key)
            print('Confidence 3: ', 100*value)
    return redirect("/")

    
if __name__ == '__main__':
    app.run(debug=True)