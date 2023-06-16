import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.python import solutions
import cv2
import blazepose_util

video_path = 'videos/00276.mp4'

model_path = 'pose_landmarker_lite.task'

mp_drawing = mp.solutions.drawing_utils

# task for pose from video
BaseOptions = mp.tasks.BaseOptions
pose = solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.8)
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the image mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
cap = cv2.VideoCapture(video_path)
# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the starting frame index to the last frame
frame_index = total_frames - 1

# Set the backward playback flag
backward_flag = cv2.CAP_PROP_POS_FRAMES
# Set the current frame index for backward playback
cap.set(backward_flag, frame_index)

while cap.isOpened():

    ret, frame = cap.read()

    if ret:
#        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Perform pose landmarking on the provided single image.
        # The pose landmarker must be created with the image mode.
#            pose_landmarker_result = landmarker.detect(mp_image)
        pose_landmarker_result = pose.process(image_rgb)
        frame_landmarks = None
        if pose_landmarker_result:
            frame_landmarks = pose_landmarker_result.pose_landmarks
        if frame_landmarks:
        #draw all landmarks
            #pose_image = blazepose_util.draw_all_landmarks_on_image(frame, pose_landmarker_result)
            pose_image = blazepose_util.draw_landmarks_on_image(frame, pose_landmarker_result)
            cv2.imshow('draw_all', pose_image)
            if cv2.waitKey(20) == ord('q'):
                break
            # draw subset
            selected_indexes = [12, 20]  # 7,11,12,13,14,15,16,23,24,25,26,27,28
            selected_landmarks = [pose_landmarker_result.pose_landmarks[0][i] for i in selected_indexes]

            #                blazepose_util.draw_subset_landmarks_on_image(mp_image, pose_landmarker_result, selected_indexes)
            # print(str(framecount)+':'+str(selected_landmarks))
            #                for l in selected_landmarks:
            #                    if l.visibility < 0.95:
            #                        print(str(framecount) + ':low confidence:' + str(l))
            #                    else:
            #                        print(str(framecount) + ':high confidence:' + str(l))
            #            else:
            #                print(str(framecount)+':'+"no landmarks")

#                if results.multi_hand_landmarks:
#                    for hand_landmarks in results.multi_hand_landmarks:
#                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#            if not frames_to_discard.detect_blur(frame):
#                cv2.imshow('pose', frame)
#            if cv2.waitKey(20) == ord('q'):
#                break
#            cv2.imshow('Video Playback', frame)

        # Decrease the frame index to move backward
        frame_index -= 1

        # Break the loop when the first frame is reached
        if frame_index < 0:
            break

        # Press 'q' to quit the playback
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

    cap.release()
    cv2.destroyAllWindows()



