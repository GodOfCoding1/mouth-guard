from pose import PoseChecker
from mouth import MouthChecker
import cv2
from time import sleep

def is_pose_correct(image_frame,mouth:MouthChecker,pose:PoseChecker):
    try:
        return True if mouth.is_pose_correct(image_frame) and pose.is_pose_correct(image_frame) else False
    except:
        raise ValueError("Some error occured")

def init(mouth:MouthChecker,pose:PoseChecker):
    input("press enter to capture default ratios")
    cap = cv2.VideoCapture(0) # 0 for webcam
    while cap.isOpened():
        _, frame = cap.read()
        is_mouth_ok = mouth.get_default_ratio(frame)
        is_pose_ok = pose.get_default_ratio(frame)
        if is_mouth_ok and is_pose_ok:
            return
        cap.release()
        cv2.destroyAllWindows()
        print("unable to detect mouth or pose or both")
        init()
 
def main():
    mouth=MouthChecker()
    pose=PoseChecker()
    init(mouth,pose)



