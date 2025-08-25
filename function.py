#utility functions
import cv2 as cv
import numpy as np
import mediapipe as mp
import os
#intializing mediapipe tools
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_hands=mp.solutions.hands
# Customize landmark style (dots on joints)
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)

# Customize connection style (lines between joints)
connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2)

#defining function to perform hand detection
def mediapipe_ley(image,model):
    #convert BGR to RGB 
    image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    results=model.process(image)
    image=cv.cvtColor(image,cv.COLOR_RGB2BGR)
    return image,results

def draw_style_landmark(image,results):
    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )
    
    cv.imshow('Hand Tracking', image)


def extract_keypoints(results):
    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks:
            rh=np.array([[res.x,res.y,res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
            return np.concatenate([rh])
        
DATA_PATH = os.path.join('/Users/abhishekbaradwaj/Desktop/sign_language/MP_Data')  
actions=np.array(['A','B','C'])
no_sequence=30
sequence_length=30