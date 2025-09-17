import tkinter as tk
from tkinter import messagebox
import csv
import tkinter as tk
from tkinter import filedialog

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import encoders
def send_email(subject, body, to_email,image_path):
    # Set up the email content
    sender_email ="test@vntechindia.com" #Replace with your email address
    sender_password = "Vntech@123"  # Replace with your email password
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = to_email
    message['Subject'] = subject

    # Attach text content
    message.attach(MIMEText(body, 'plain'))

    # Attach the image
    with open(image_path, 'rb') as image_file:
        image = MIMEImage(image_file.read())
        image.add_header('Content-Disposition', 'attachment', filename="1.png")
        message.attach(image)

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.hostinger.com', 587) as server:
        #server.starttls()
        # Login to your email account
        server.login(sender_email, sender_password)
        # Send the email
        server.sendmail(sender_email, to_email, message.as_string())
        print("done")
def exit_application():
    root.destroy()

def show_home():
    home_frame.pack()
    registration_frame.pack_forget()
    login_frame.pack_forget()
    welcome_frame.pack_forget()

def show_registration():
    home_frame.pack_forget()
    registration_frame.pack()
    login_frame.pack_forget()
    welcome_frame.pack_forget()

def register_user():
    global name
    name = name_entry.get()
    email = email_entry.get()
    username = username_entry.get()
    password = password_entry.get()

    with open('users.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, email, username, password])

    messagebox.showinfo("Registration", "User registered successfully!")

def show_login():
    home_frame.pack_forget()
    registration_frame.pack_forget()
    login_frame.pack()
    welcome_frame.pack_forget()

def login_user():
    global  username
    username = login_username_entry.get()
    password = login_password_entry.get()

    with open('users.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[2] == username and row[3] == password:
                messagebox.showinfo("Login", "Login successful!")
                show_welcome()
                return

    messagebox.showerror("Login", "Invalid username or password!")

def show_welcome():
    home_frame.pack_forget()
    registration_frame.pack_forget()
    login_frame.pack_forget()
    welcome_frame.pack()
    hello_button.pack()

from tkinter import *
import tkinter
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2


# main = tkinter.Tk()
# main.title("Student Attentiveness Monitoring")
# main.geometry("500x400")

def EAR(studenteye):#Eye aspect ratio
    point1 = dist.euclidean(studenteye[1], studenteye[5])
    point2 = dist.euclidean(studenteye[2], studenteye[4])
    # compute the euclidean distance between the horizontal
    distance = dist.euclidean(studenteye[0], studenteye[3])
    # compute the eye aspect ratio
    ear_aspect_ratio = (point1 + point2) / (2.0 * distance)
    return ear_aspect_ratio

def MAR(studentmouth):#Mouth aspect ratio
    # compute the euclidean distances between the horizontal
    point   = dist.euclidean(studentmouth[0], studentmouth[6])
    # compute the euclidean distances between the vertical
    point1  = dist.euclidean(studentmouth[2], studentmouth[10])
    point2  = dist.euclidean(studentmouth[4], studentmouth[8])
    # taking average
    Ypoint   = (point1+point2)/2.0
    # compute mouth aspect ratio
    mouth_aspect_ratio = Ypoint/point
    return mouth_aspect_ratio
    
def startMonitoring():
    # pathlabel.config(text="          Webcam Connected Successfully")
    webcamera = cv2.VideoCapture(0)
    svm_predictor_path = 'SVMclassifier.dat'
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 10
    MOU_AR_THRESH = 0.75

    COUNTER = 0
    yawnStatus = False
    yawns = 0
    svm_detector = dlib.get_frontal_face_detector()
    svm_predictor = dlib.shape_predictor(svm_predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    root_dir = os.getcwd()
    # Load Face Detection Model
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    # Load Anti-Spoofing Model graph
    json_file = open('antispoofing_models/antispoofing_model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load antispoofing model weights 
    model.load_weights('antispoofing_models/antispoofing_model.h5')
    print("Model loaded from disk")
    while True:
        ret, frame = webcamera.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_yawn_status = yawnStatus
        rects = svm_detector(gray, 0)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:  
            face = frame[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
            # resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(resized_face)[0]
            print(preds)
            if preds> 0.5:
                label = 'spoof'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 0, 255), 2)
            else:
                label = 'real'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                (0, 255, 0), 2)
        #cv2.imshow('frame', frame)
        for rect in rects:
            shape = svm_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            mouEAR = MAR(mouth)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                cv2.putText(frame, "Eyes Closed ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "STUDENT DISTRACTION ALERT", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite("1.png",frame)
                    import winsound
                    frequency = 2500  # Set Frequency To 2500 Hertz
                    duration = 1000  # Set Duration To 1000 ms == 1 second
                    winsound.Beep(frequency, duration)
                    subject = username+" not paying attention"
                    body = "This is a student not paying attention sent via Python."
                    to_email = "srijadevathi@gmail.com"  # Replace with the recipient's email address
                    image_path="1.png"
                    send_email(subject, body, to_email, image_path)
            else:
                COUNTER = 0
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if mouEAR > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning, DROWSINESS ALERT! ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                import winsound
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                yawnStatus = True
                output_text = "Yawn Count: " + str(yawns + 1)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
                cv2.imwrite("1.png",frame)
                import winsound
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                subject = username+" student not paying attention"
                body = "This is a student not paying attention sent via Python."
                to_email = "srijadevathi@gmail.com"  # Replace with the recipient's email address
                image_path="1.png"
                send_email(subject, body, to_email, image_path)
            else:
                yawnStatus = False
            if prev_yawn_status == True and yawnStatus == False:
                yawns+=1
            cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame,"Visual Behaviour & Machine Learning Drowsiness Detection @ Drowsiness",(370,470),cv2.FONT_HERSHEY_COMPLEX,0.6,(153,51,102),1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    webcamera.release()    

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        print("Selected File:", file_path)
        return file_path



def uploadVideo():
    # pathlabel.config(text="          Webcam Connected Successfully")
    file_path= open_file_dialog()
    webcamera = cv2.VideoCapture(file_path)
    svm_predictor_path = 'SVMclassifier.dat'
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 10
    MOU_AR_THRESH = 0.75

    COUNTER = 0
    yawnStatus = False
    yawns = 0
    svm_detector = dlib.get_frontal_face_detector()
    svm_predictor = dlib.shape_predictor(svm_predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    while True:
        
        ret, frame = webcamera.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_yawn_status = yawnStatus
        rects = svm_detector(gray, 0)
        for rect in rects:
            shape = svm_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            mouEAR = MAR(mouth)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                cv2.putText(frame, "Eyes Closed ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "STUDENT DISTRACTION ALERT", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    import winsound
                    frequency = 2500  # Set Frequency To 2500 Hertz
                    duration = 1000  # Set Duration To 1000 ms == 1 second
                    winsound.Beep(frequency, duration)
                    cv2.imwrite("1.png",frame)
                    import winsound
                    frequency = 2500  # Set Frequency To 2500 Hertz
                    duration = 1000  # Set Duration To 1000 ms == 1 second
                    winsound.Beep(frequency, duration)
                    subject = username+" student not paying attention"
                    body = "This is a student not paying attention sent via Python."
                    to_email = "srijadevathi@gmail.com"  # Replace with the recipient's email address
                    image_path="1.png"
                    send_email(subject, body, to_email, image_path)
            else:
                COUNTER = 0
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if mouEAR > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning, DROWSINESS ALERT! ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawnStatus = True
                output_text = "Yawn Count: " + str(yawns + 1)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
                import winsound
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                cv2.imwrite("1.png",frame)
                subject = username+" student not paying attention"
                body = "This is a student not paying attention sent via Python."
                to_email = "srijadevathi@gmail.com"
                image_path="1.png"
                send_email(subject, body, to_email, image_path)
            else:
                yawnStatus = False
            if prev_yawn_status == True and yawnStatus == False:
                yawns+=1
            cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame,"Visual Behaviour & Machine Learning Drowsiness Detection @ Drowsiness",(370,470),cv2.FONT_HERSHEY_COMPLEX,0.6,(153,51,102),1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    webcamera.release()   
  

# font = ('times', 16, 'bold')
# title = Label(main, text='Student Behaviour Monitoring System using Visual\n               Behaviour and Machine Learning',anchor=W, justify=LEFT)
# title.config(bg='black', fg='white')  
# title.config(font=font)           
# title.config(height=3, width=120)       
# title.place(x=0,y=5)


# font1 = ('times', 14, 'bold')
# upload = Button(main, text="Start Behaviour Monitoring Using Webcam", command=startMonitoring)
# upload.place(x=50,y=200)
# upload.config(font=font1)  

# pathlabel = Label(main)
# pathlabel.config(bg='green', fg='white')  
# pathlabel.config(font=font1)           
# pathlabel.place(x=50,y=250)


# main.config(bg='green')
# main.mainloop()


# Create main window
root = tk.Tk()
root.title("Students activity monitoring system")
root.geometry("500x400")
root.configure(bg="green")
# Frames
home_frame = tk.Frame(root)
registration_frame = tk.Frame(root)
login_frame = tk.Frame(root)
welcome_frame = tk.Frame(root)
home_frame.configure(bg="green")
# Home Screen
home_label = tk.Label(home_frame, text="Welcome to student activity monitoring system", font=("Helvetica", 14))
home_label.pack(pady=20)

registration_button = tk.Button(home_frame, text="Register", command=show_registration)
registration_button.pack(pady=40)

login_button = tk.Button(home_frame, text="Login", command=show_login)
login_button.pack(pady=40)

# Registration Screen
name_label = tk.Label(registration_frame, text="Name:")
name_label.pack(pady=5)
name_entry = tk.Entry(registration_frame)
name_entry.pack(pady=5)

email_label = tk.Label(registration_frame, text="Email:")
email_label.pack(pady=5)
email_entry = tk.Entry(registration_frame)
email_entry.pack(pady=5)

username_label = tk.Label(registration_frame, text="Username:")
username_label.pack(pady=5)
username_entry = tk.Entry(registration_frame)
username_entry.pack(pady=5)

password_label = tk.Label(registration_frame, text="Password:")
password_label.pack(pady=5)
password_entry = tk.Entry(registration_frame, show="*")
password_entry.pack(pady=5)

register_button = tk.Button(registration_frame, text="Register", command=register_user)
register_button.pack(pady=40)

back_to_home_button_reg = tk.Button(registration_frame, text="Back to Home", command=show_home)
back_to_home_button_reg.pack(pady=40)

# Login Screen
login_username_label = tk.Label(login_frame, text="Username:")
login_username_label.pack(pady=5)
login_username_entry = tk.Entry(login_frame)
login_username_entry.pack(pady=5)

login_password_label = tk.Label(login_frame, text="Password:")
login_password_label.pack(pady=5)
login_password_entry = tk.Entry(login_frame, show="*")
login_password_entry.pack(pady=5)

login_button = tk.Button(login_frame, text="Login", command=login_user)
login_button.pack(pady=40)

back_to_home_button_login = tk.Button(login_frame, text="Back to Home", command=show_home)
back_to_home_button_login.pack(pady=40)

# Welcome Screen
hello_label = tk.Label(welcome_frame, text="Students activity monitoring system", font=("Helvetica", 14))
hello_label.pack(pady=20)

hello_button = tk.Button(welcome_frame, text="click for live monitoring ", command=startMonitoring)
hello_button.pack(pady=40)

hello_button = tk.Button(welcome_frame, text="click for analysis in video ", command=uploadVideo)
hello_button.pack(pady=40) 

# exit_button = tk.Button(root, text="Exit", command=exit_application)
# exit_button.pack(pady=40)


back_to_home_button_welcome = tk.Button(welcome_frame, text="Back to Home", command=show_home)
back_to_home_button_welcome.pack(pady=40)


# Initial view
show_home()

# Start the main loop
root.mainloop()
