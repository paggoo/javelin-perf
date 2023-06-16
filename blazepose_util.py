import numpy as np
import cv2
from mediapipe.python import solutions
from mediapipe.framework.formats import landmark_pb2 as land


LOW_PRESENCE_THRESHOLD = 0.8
LOW_VISIBILITY_THRESHOLD = 0.8
LOW_PRESENCE_AVG_THRESHOLD = 0.97
LOW_VISIBILITY_AVG_THRESHOLD = 0.97

circle_border_radius = 5
thickness = 3
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

def draw_circles(annotated_image, list):
    for circle in list:
        circle_x = circle[0]
        circle_y = circle[1]
        cv2.circle(annotated_image, circle_x, circle_y, circle_border_radius, WHITE_COLOR, thickness)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks

    annotated_image = np.copy(rgb_image)

    # Draw the pose landmarks.
    pose_landmarks_proto = land.NormalizedLandmarkList()
    if pose_landmarks_list:
        for landmark in pose_landmarks_list:
            pose_landmarks_proto.landmark.extend(land.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z))
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
        #      solutions.pose.POSE_CONNECTIONS.difference((0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10)),
          solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def draw_all_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = land.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          land.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def draw_subset_landmarks_on_image(rgb_image, detection_result, indexes):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = land.NormalizedLandmarkList()
    selected_landmarks = []
    for id in indexes:
        selected_landmarks.extend(pose_landmarks[id])
    pose_landmarks_proto.landmark.extend([
      land.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in selected_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def is_landmark_low_presence(landmark):
    return landmark.presence < LOW_PRESENCE_THRESHOLD


def remove_low_presence_landmarks_from_pose(pose):
    for landmark in pose:
        if is_landmark_low_presence(landmark):
            pose.remove(landmark)
    return pose


def is_pose_low_presence(pose):
    s = 0
    for landmark in pose[0]:
        s += landmark.presence
    return s>len(pose)*LOW_PRESENCE_AVG_THRESHOLD


def is_landmark_low_visibility(landmark):
    return landmark.visibility < LOW_VISIBILITY_THRESHOLD


def remove_low_visibility_landmarks_from_pose(pose):
    for landmark in pose:
        if is_landmark_low_visibility(landmark):
            pose.remove(landmark)
    return pose


def is_pose_low_visibility(pose):
    s = 0
    for landmark in pose[0]:
        s += landmark.visibility
    return s<len(pose[0])*LOW_VISIBILITY_AVG_THRESHOLD



#    cv2.circle(annotated_image, (int(pose_landmarks.x * annotated_image.shape[1]), int(pose_landmarks.y * annotated_image.shape[0])),
#               circle_border_radius, WHITE_COLOR, thickness)
