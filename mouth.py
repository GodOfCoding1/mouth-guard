# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from cv2 import typing

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def get_lip_diff(faces_landmarks):
    upper_mean=sum(faces_landmarks[0][i].y for i in range(11,14) )
    lower_mean=sum(faces_landmarks[0][i].y for i in range(14,17))
    total_lip_height=abs(faces_landmarks[0][11].y-faces_landmarks[0][16].y)
    return abs(upper_mean-lower_mean)/total_lip_height

def get_default_ratio(image_frame:typing.MatLike,model):
    input("< press enter to take picture for pose measurements (if face is not visible this will repeat)>")
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

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

#use camera
cap = cv2.VideoCapture(0)

#first time running loop
closed_relative_difference=None
while cap.isOpened():
    if closed_relative_difference:
        break
    input("< press enter to take picture for closed mouth measurements (if face is not visible this will repeat)>")
    success, image = cap.read()
    # STEP 3: Load the input image.
    # image = mp.Image.create_from_file("image.png")
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)
    if len(detection_result.face_landmarks):
        closed_relative_difference=get_lip_diff(detection_result.face_landmarks)
        print("took measurements")
    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    # cv2.imshow("my image",annotated_image)
    if cv2.waitKey(100) == ord('q'):
        break

input("press enter to start")
#always running loop
while cap.isOpened():
        success, image = cap.read()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(image)
        state="closed"
       
        if len(detection_result.face_landmarks):
            current_diff=get_lip_diff(detection_result.face_landmarks)
            if current_diff>closed_relative_difference*1.05:
                print("mouth is open")
            else:
                print("mouth is closed")
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow("my face",annotated_image)
        if cv2.waitKey(100) == ord('q'):
            break
        # checks every 1 second
        sleep(1)

class PoseChecker():
    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                              output_face_blendshapes=True,
                                              output_facial_transformation_matrixes=True,
                                              num_faces=1)
        self.model=vision.FaceLandmarker.create_from_options(options)
        self.default_ratio=None
        self.error_counter=0
        self.error_threshold=10

    def process_ratio_from_image(self,image_frame:typing.MatLike):
        # Recolor image to RGB
        image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.model.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            self.error_counter-=1
            return get_mouth_shoulder_ratio(landmarks)
        except:
            self.error_counter+=1
            if self.error_counter>self.error_threshold:
                raise ValueError(f"Some error occured consecutively for {self.error_threshold} times")
        
    def is_pose_correct(self,image_frame:typing.MatLike):
        if self.default_ratio:
            current_ratio=self.process_ratio_from_image(image_frame)
            return False if current_ratio<self.default_ratio*0.8 else True
        else:
            raise ValueError("You have to call getDeafultRatio before calling is_pose_correct")