import os
from app import app
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False


model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    lst=[]
    #print(filename)
    img = cv2.imread('static/uploads/'+filename)
    #cv2.imshow(img)
    window = np.zeros((940, 940, 3), dtype="uint8")

    frm = cv2.flip(img, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    cv2.putText(window, "Yoga Pose Detector", (220, 50), cv2.FONT_ITALIC, 1.0, (255, 255, 0), 2)

    frm = cv2.blur(frm, (4, 4))
    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        lst = np.array(lst).reshape(1, -1)

        p = model.predict(lst)
        pred = label[np.argmax(p)]
        if p[0][np.argmax(p)] > 0.75:
            cv2.putText(window, "Pose :" + pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)


        else:
            cv2.putText(window, "Asana is either wrong not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)

    else:
        cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                           connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                           landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

    frm = cv2.resize(frm, (640, 480))

    # frm = detector.findPose(frm)
    # lmlist, boxinfo = detector.findPosition(frm, bboxWithHands=True)

    gray = cv2.cvtColor(frm, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frm, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frm, (xA, yA), (xB, yB), (0, 255, 0), 2)
    window[420:900, 170:810, :] = frm
    print('KSDNKASD')
    cv2.imwrite('output.jpg',window)
    return redirect(url_for('static/uploads/output.jpg'), code=301)


if __name__ == "__main__":
    app.run()