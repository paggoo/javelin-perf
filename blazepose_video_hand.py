
#video_path = 'hand.webm'
video_path = 'videos/00099.mp4'



import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

video_cap = cv2.VideoCapture(video_path)
while True:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = video_cap.read()
    if success:
#        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        image = frame
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results_pose = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('BlazePose Output', image)
        if cv2.waitKey(20) == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()

