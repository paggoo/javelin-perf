import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python import solutions
from mediapipe.framework.formats import landmark_pb2

import blazepose_util
import frames_to_discard

model_path = 'pose_landmarker_lite.task'

# task for pose from video
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the image mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)


def start_frame(video_path, execution_count=0): #detect first pose #remove first frames due to blur
    video = cv2.VideoCapture(video_path)
    with PoseLandmarker.create_from_options(options) as landmarker:
        framecount = -1
        while True:
            # `success` is a boolean and `frame` contains the next video frame
            success, frame = video.read()
            if success:             #frame read
                framecount += 1
                if not frames_to_discard.detect_blur(frame):            #frame acceptable
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    pose_landmarker_result = landmarker.detect(mp_image)
                    if pose_landmarker_result.pose_landmarks:                       #pose detected
                        # reject frames indicating person size smaller threshold to sort out background people
                        # careful: analysis between smallest proband video 94 and biggest background person video 277 indicates this method might not be safe
                        # alternative would be multi-person detection
#                        cv2.imshow('start'+video_path, frame)
#                        cv2.waitKey(0)
#                        print(max (pose_landmarker_result.pose_world_landmarks[0][31].y, pose_landmarker_result.pose_world_landmarks[0][32].y)
#                            - pose_landmarker_result.pose_world_landmarks[0][6].y)
#                        print(max (pose_landmarker_result.pose_landmarks[0][31].y, pose_landmarker_result.pose_landmarks[0][32].y)
#                            - pose_landmarker_result.pose_landmarks[0][6].y)
                        if execution_count == 1:
                            # print('start detected')
                            # draw full pose for analysis
                            frame = blazepose_util.draw_all_landmarks_on_image(frame, pose_landmarker_result)
                            # get nose coordinates @start
                            start = pose_landmarker_result.pose_landmarks
                            #                            print(start) #watch out for the z coordinate
                            x_start = int(start[0][00].x * mp_image.width)

                            # Define the start and end points for the vertical line
                            start_point = (x_start, 0)  # starting from the top
                            end_point = (x_start, mp_image.height)  # ending at the bottom

                            # Set the line color and thickness
                            color = (0, 255, 0)  # Green color in BGR format
                            thickness = 2

                            # Draw the vertical line on the image
                            cv2.line(frame, start_point, end_point, color, thickness)
                            cv2.imwrite(video_path.split('.')[0] + '_start.png', frame)  #file must be written by calling function. NOT HERE!
                        elif execution_count == 0:
                            if (            # size due to background people and low_presence detection
                                    (max (pose_landmarker_result.pose_landmarks[0][31].y, pose_landmarker_result.pose_landmarks[0][32].y)
                                     - pose_landmarker_result.pose_landmarks[0][6].y) > 0.34 ) \
                                    and not blazepose_util.is_pose_low_visibility(pose_landmarker_result.pose_landmarks):

                                #print('start detected')
                                #draw full pose for analysis
                                frame = blazepose_util.draw_all_landmarks_on_image(frame, pose_landmarker_result)
                                # get nose coordinates @start
                                start = pose_landmarker_result.pose_landmarks
    #                            print(start) #watch out for the z coordinate
                                x_start = int(start[0][00].x * mp_image.width)


                                # Define the start and end points for the vertical line
                                start_point = (x_start, 0)  # starting from the top
                                end_point = (x_start, mp_image.height)  # ending at the bottom

                                # Set the line color and thickness
                                color = (0, 255, 0)  # Green color in BGR format
                                thickness = 2

                                # Draw the vertical line on the image
                                cv2.line(frame, start_point, end_point, color, thickness)
                                cv2.imwrite(video_path.split('.')[0] + '_start.png', frame)
    #                            cv2.imshow('start'+video_path, frame)
    #                            cv2.waitKey(0)
                                break
            else: #video end reached
                if execution_count == 0:
                   start_frame(video_path, 1) #second_chance
                else:
                    print("no start detected")
                    break
                    return -1
    # we also need to close the video and destroy all Windows
    video.release()
    cv2.destroyAllWindows()
    if video_path in file_data:
        file_data[video_path]["start_frame"] = framecount
    else:
        file_data[video_path] = {}
        file_data[video_path]["start_frame"] = framecount
    return framecount


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


def release_frame(video_path): #(opening fingers not recognizable, hand-ear-distance differs from probands motion attempts)
    #when throwing hand_x > shoulder_x AND (shoulderx-handx)/time > threshold
    #hand schnellt an schulter vorbei
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    with PoseLandmarker.create_from_options(options) as landmarker:
        if video_path in file_data and 'start_frame' in file_data[video_path]:
            framecount = file_data[video_path]['start_frame']
        else:
            framecount = file_data[video_path]["start_frame"] = start_frame(video_path)
#        print('start_frame:'+str(framecount))
        # Read the frame at the desired frame number
        while True:
            video.set(cv2.CAP_PROP_POS_FRAMES, framecount)
            success, frame = video.read()
            success_next, frame_next = video.read()
            # Check if the frame was retrieved successfully
            if success and success_next:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                mp_image_next = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_next)
                pose_landmarker_result = landmarker.detect(mp_image)
                pose_landmarker_result_next = landmarker.detect(mp_image_next)
                if pose_landmarker_result.pose_landmarks and pose_landmarker_result_next.pose_landmarks:
                    #draw poses for anaylsis and glue images next to each other
                    frame = blazepose_util.draw_all_landmarks_on_image(frame, pose_landmarker_result)
                    frame_next = blazepose_util.draw_all_landmarks_on_image(frame_next, pose_landmarker_result_next)
                    # resize images
                    original_height, original_width = frame.shape[:2]
                    original_height, original_width = frame_next.shape[:2]  #they have same size so dont worry
                    # Calculate the new dimensions for scaling to 50%
                    new_height = int(original_height * 0.5)
                    new_width = int(original_width * 0.5)
                    # Resize the image
                    resized_frame = cv2.resize(frame, (new_width, new_height))
                    resized_frame_next = cv2.resize(frame_next, (new_width, new_height))
                    before_after_image = cv2.hconcat([resized_frame, resized_frame_next])
                    cv2.imshow('end:'+video_path, before_after_image)
                    cv2.waitKey(0)
                    shoulder = pose_landmarker_result.pose_landmarks[0][12]
#                    print(shoulder)
                    index_finger = pose_landmarker_result.pose_landmarks[0][20]
#                    print(index_finger)
                    hand_before_shoulder_x = index_finger.x-shoulder.x #changes from neg to pos when passing shoulder
#                    print('hand_before_shoulder_x:'+str(hand_before_shoulder_x))

                    shoulder_next = pose_landmarker_result_next.pose_landmarks[0][12]
#                    print(shoulder_next)
                    index_finger_next = pose_landmarker_result_next.pose_landmarks[0][20]
#                    print(index_finger_next)
                    hand_before_shoulder_next_x = index_finger_next.x-shoulder_next.x
#                    print('hand_before_shoulder_next_x:'+str(hand_before_shoulder_next_x))

                    throw_x_distance_at_frame = (hand_before_shoulder_next_x - hand_before_shoulder_x) #distance first, then speed
#                    print('hand_shoulder_distance:'+str(throw_x_distance_at_frame))
                    throw_x_velocity_at_frame = (index_finger_next.x - index_finger.x)/(1/fps)
                    shoulder_x_velocity_at_frame = (shoulder_next.x - shoulder.x)/(1/fps)
                    print('hand_speed:'+str(throw_x_velocity_at_frame))
                    print('shoulder_speed:'+str(shoulder_x_velocity_at_frame))
                    if hand_before_shoulder_x <= 0 and hand_before_shoulder_next_x > 0 :
                        print('hand_before_shoulder! release_frame: '+str(framecount))
                        print('hand_next: '+str(index_finger_next))
                        print('shoulder_next:' +str(shoulder_next))
                        print('hand_before_shoulder_next_x:' + str(hand_before_shoulder_next_x))
                        if throw_x_velocity_at_frame > 1:  # alternativ: ellbogen durchgestreckt

    #                    cv2.imshow('frame',frame)
    #                    cv2.waitKey(0)
    #                    cv2.imshow('frame_next',frame_next)
    #                    cv2.waitKey(0)
    #                    if throw_x_velocity_at_frame > 4.2 and hand_before_shoulder_next_x < 0:
                            # get nose coordinates @release
                            x_release = int(pose_landmarker_result.pose_landmarks[0][00].x * mp_image.width)

                            # Define the start and end points for the vertical line
                            start_point = (x_release, 0)  # starting from the top
                            end_point = (x_release, mp_image.height)  # ending at the bottom

                            # Set the line color and thickness
                            color = (0, 0, 255)  # Red color in BGR format
                            thickness = 2

                            # Draw the vertical line on the image
     #                       cv2.line(frame, start_point, end_point, color, thickness)
    #                        cv2.imwrite(video_path.split('.')[0]+'_release.png', mp_image)
    #                        cv2.imshow('end'+video_path, mp_image)
    #                        cv2.waitKey(0)
                            return framecount
                framecount += 1
#                print(framecount)
            else:
                break
        # get nose coordinates @release
#        x_release = int(pose_landmarker_result.pose_landmarks[0][00].x * mp_image.width)
    video.release()
    cv2.destroyAllWindows()
    if video_path in file_data:
        file_data[video_path]["release_frame"] = framecount
    else:
        file_data[video_path] = {}
        file_data[video_path]["release_frame"] = framecount
    return framecount


def release_height(video_path):
    video = cv2.VideoCapture(video_path)
    with PoseLandmarker.create_from_options(options) as landmarker:
        if video_path in file_data and "release_frame" in file_data[video_path]:
            var = file_data[video_path]["release_frame"]
        else:
            var = release_frame(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, var-1)
        # Read the frame at the desired frame number
        ret, frame = video.read()
        # Check if the frame was retrieved successfully
        if not ret:
            print("Failed to retrieve the frame.")
            exit()
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_landmarker_result = landmarker.detect(mp_image)
            if pose_landmarker_result.pose_landmarks:
                #get ground_y - hand_y coordinates @release
                #using right foot, right hand due to different distance from camera results in different pixel height leg far from camera
                right_hand = int(pose_landmarker_result.pose_landmarks[0][16].y*mp_image.width)
                right_foot = int(pose_landmarker_result.pose_landmarks[0][32].y * mp_image.width)
                return right_foot-right_hand


# due to fluctuation measure at least 5 frames
def runway_distance(video_path):
    video = cv2.VideoCapture(video_path)
    x_release = 0
    x_start = 0
    with PoseLandmarker.create_from_options(options) as landmarker:
        var = file_data[video_path]["start_frame"]
        print('variable'+str(var))
        video.set(cv2.CAP_PROP_POS_FRAMES, var)
        # Read the frame at the desired frame number
        ret, frame = video.read()
        # Check if the frame was retrieved successfully
        if not ret:
            print("Failed to retrieve the frame.")
            exit()
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_landmarker_result = landmarker.detect(mp_image)
            if pose_landmarker_result.pose_landmarks:
                #get nose coordinates @run_start
                x_start = int(pose_landmarker_result.pose_landmarks[0][00].x*mp_image.width)

                # Define the start and end points for the vertical line
                start_point = (x_start, 0)  # starting from the top
                end_point = (x_start, mp_image.height)  # ending at the bottom

                # Set the line color and thickness
                color = (0, 255, 0)  # Green color in BGR format
                thickness = 2

                # Draw the vertical line on the image
                cv2.line(frame, start_point, end_point, color, thickness)

                cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return x_release - x_start


def runway_duration(video_path): #evtl Metadaten in die Excel Liste eintragen oder dictionary um linearen Durchlauf des videos zu sparen
    print("TODO")
    return "lookup stop - lookup start"


def runway_avg_speed(video_path):
    print("TODO")
    return "lookup duration/ lookup fps"


def runway_speed_at_release(video_path):
    print("TODO")
    return "lookup release, take 2-4 frames / lookup fps"


def runway_speed_at_action(video_path, frame_id):
    print("TODO")
    return "if frame_id < lookup stop and frame_id > lookup start then take 2-4 frames / lookup fps"


def relative_to_absolute_x_distance(video_path):
    print("TODO")
    print('take pose from middle of frames to sort start unsharpness# better from frame with highest sharpness')
    print('stretch person (sum low leg+ up leg + trunk height + eye height + scheitel about same as eye to opposite mouth corner')
    print('take person real height from table')


def avg_runway_speed(video_path):
    print("TODO")
    print('distance by time between releaseframe-startframe')


video_path = 'videos/00023.mp4'
image_path = 'blur1.png'
image = cv2.imread(image_path)
#print('runway_start_frame:'+str(start_frame('videos/00099.mp4')))
#print('runway_distance:'+str(runway_distance(video_path)))
#print('release_frame:'+str(release_frame(video_path)))
#print('release_height:', str(release_height(video_path)))
#print(frames_to_discard.detect_blur(image))

path = 'videos'
#for file_name in os.listdir(path):
#    if file_name.endswith('.mp4'):
#        file_path = os.path.join(path, file_name)
#        start_frame(file_path)

#baue dictionary um features pro datei festzuhalten
file_data = {}

"""
# Iterate over the file names
for file_name in os.listdir(path):
    if file_name.endswith('.mp4'):
        # the files are in the path directory
        file_path = os.path.join(path, file_name)

        #open feature dictionary of current file
        file_data[file_path] = {}

        # store values in inner dictionary
        #start
        print('trying:'+file_path)
        start = start_frame(file_path)
        file_data[file_path]["start_frame"] = start
        #release
        release = release_frame(file_path)
        file_data[file_path]["release_frame"] = release
        print(file_path+':'+str(file_data[file_path]))

#                "runway_avg": values[2],
#                "runway_max": values[3],
#                "runway_release": values[4],
#                "release_height_release_speed": values[5]

# Print the resulting dictionary
print(file_data.values())
"""
#start_frame("videos/00005.mp4")
release_frame("videos/00270.mp4")