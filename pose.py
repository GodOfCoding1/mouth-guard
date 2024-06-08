
# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from time import sleep
from cv2 import typing

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_landmarks(image:typing.MatLike,results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
    cv2.imshow('Mediapipe Feed', image)

def get_mouth_shoulder_ratio(landmarks) -> float:
    mouth_left=landmarks[10]
    mouth_right=landmarks[9]
    mouth_height=(mouth_left.y+mouth_right.y)/2
    shoulder_left=landmarks[12]
    shoulder_right=landmarks[11]
    shoulder_height=(shoulder_left.y+shoulder_right.y)/2
    shoulder_span=abs(shoulder_left.x-shoulder_right.x)
    return abs(mouth_height-shoulder_height)/shoulder_span

def is_pose_correct(image_frame:typing.MatLike,model,default_ratio):
    # Recolor image to RGB
    image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        current_ratio=get_mouth_shoulder_ratio(landmarks)
        if current_ratio<default_ratio*0.8:
            return False
        else:
            return True
    except:
        raise ValueError("error detecting body")
    
def get_default_ratio(image_frame:typing.MatLike,model):
    input("< press enter to take picture for pose measurements (if face & shoulders is not visible this will repeat)>")
    # Recolor image to RGB
    image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        return get_mouth_shoulder_ratio(landmarks)
    except:
        pass
    


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:# Setup mediapipe instance
    cap = cv2.VideoCapture(0)
    #first time running loop
    correct_ratio=None
    while cap.isOpened():
        if correct_ratio:
            break
        _, frame = cap.read()
        correct_ratio=get_default_ratio(image_frame=frame,model=pose)
        print(correct_ratio)
        
    input("press enter to start")
    while cap.isOpened():
        _, frame = cap.read()
        try:
            if is_pose_correct(image_frame=frame,model=pose,default_ratio=correct_ratio):
                print("pose is correct")
            else:
                print("pose is wrong")
        except:
            print("erroorrr")
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        sleep(2)

    cap.release()
    cv2.destroyAllWindows()