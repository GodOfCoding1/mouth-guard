
# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from time import sleep

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_mouth_shoulder_ratio(landmarks):
    mouth_left=landmarks[10]
    mouth_right=landmarks[9]
    mouth_height=(mouth_left.y+mouth_right.y)/2
    shoulder_left=landmarks[12]
    shoulder_right=landmarks[11]
    shoulder_height=(shoulder_left.y+shoulder_right.y)/2
    shoulder_span=abs(shoulder_left.x-shoulder_right.x)
    return abs(mouth_height-shoulder_height)/shoulder_span
    

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture(0)
    #first time running loop
    correct_ratio=None
    while cap.isOpened():
        if correct_ratio:
            break
        input("< press enter to take picture for pose measurements (if face & shoulders is not visible this will repeat)>")
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            correct_ratio=get_mouth_shoulder_ratio(landmarks)
        except:
            pass

    input("press enter to start")
    ## Setup mediapipe instance
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            current_ratio=get_mouth_shoulder_ratio(landmarks)
            print(current_ratio,correct_ratio)
            if current_ratio<correct_ratio*0.8:
                print("pose is wrong")
            else:
                print("pose is correct")
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        sleep(2)

    cap.release()
    cv2.destroyAllWindows()