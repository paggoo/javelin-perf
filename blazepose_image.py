import torch
import torch.nn as nn


#load image
import cv2

image_path = 'pic.png'
image = cv2.imread(image_path)

# Preprocess the image if required (e.g., resizing, normalization)

# Convert the image to the required format (e.g., BGR for OpenPose)

# Pass the preprocessed image to OpenPose for pose estimation


# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # Forward pass through RNN layer
        out, hidden = self.rnn(x, hidden)

        # Reshape the output for the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)

        # Forward pass through the fully connected layer
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden


# Define the input data
input_size = 10  # Input size of each timestep
hidden_size = 32  # Size of the hidden state
output_size = 2  # Output size

# Create an instance of the RNN model
rnn = RNN(input_size, hidden_size, output_size)

# Create a random input tensor (batch_size, sequence_length, input_size)
batch_size = 16
sequence_length = 20
input_data = torch.randn(batch_size, sequence_length, input_size)

# Forward pass through the RNN model
output, hidden = rnn(input_data)

# Print the output shape
print("Output shape:", output.shape)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'pose_landmarker_lite.task'

# task for pose from image
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with PoseLandmarker.create_from_options(options) as landmarker:
# The landmarker is initialized. Use it here.
# ...


# Load the input image from an image file.
    mp_image = mp.Image.create_from_file("pic.png")

# Load the input image from a numpy array.
#mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

# Perform pose landmarking on the provided single image.
# The pose landmarker must be created with the image mode.
    pose_landmarker_result = landmarker.detect(mp_image)

from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe.python import solutions

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

import cv2

img = cv2.imread('pic.png')
#cv2.imshow('bla', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

pose_image = draw_landmarks_on_image(img, pose_landmarker_result)
cv2.imshow('pose', pose_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
