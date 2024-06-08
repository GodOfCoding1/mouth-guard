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