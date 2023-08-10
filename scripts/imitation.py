#!/usr/bin/python3

import service
import cv2
import roslib
import rospy
from std_msgs.msg import Int32MultiArray
from collections import deque
import numpy as np


def main(color=(224, 255, 255)):

    config_path = roslib.packages.get_pkg_dir('head_pose_imitation') + "/config"

    fd = service.UltraLightFaceDetecion(config_path + "/weights/RFB-320.tflite",
                                       conf_threshold=0.95)

    fa = service.DepthFacialLandmarks(config_path + "/weights/sparse_face.tflite")

    # Constants for mapping detected angles space to motor commands space
    pitch_range = [-30, 30]
    roll_range = [-50, 50]
    yaw_range = [-90, 90]
    pitch_command_range = [180, 110]
    roll_command_range = [100, 190]
    yaw_command_range = [90, 210]

    cap = cv2.VideoCapture(0)
    command_array = Int32MultiArray()
    command_array.data = [0] * 23
    command_array.data[20:23] = [145] * 3

    rospy.init_node('head_pose_imitation_node', anonymous=True)
    publisher = rospy.Publisher('jointdata/qc', Int32MultiArray, queue_size=100)

    euler_queue = deque(maxlen=5)  # Moving average queue for euler angles

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # face detection
        boxes, scores = fd.inference(frame)

        # raw copy for reconstruction
        feed = frame.copy()
        if len(boxes) > 0:
            bbox = [boxes[0]]
            for results in fa.get_landmarks(feed, bbox):
                euler = service.pose(frame, results, color)
                euler_queue.append(euler)  # Add euler angles to the moving average queue

            if len(euler_queue) > 0:
                averaged_euler = np.mean(euler_queue, axis=0)  # Calculate the average of euler angles
                print(f"Pitch: {averaged_euler[0]}; Yaw: {averaged_euler[1]}; Roll: {averaged_euler[2]};")
                pitch = pitch_command_range[0] + (pitch_command_range[1] - pitch_command_range[0]) * ((-averaged_euler[0] - pitch_range[0]) / (pitch_range[1] - pitch_range[0]))
                roll = roll_command_range[0] + (roll_command_range[1] - roll_command_range[0]) * ((-averaged_euler[2] - (roll_range[0])) / (roll_range[1] - (roll_range[0])))
                yaw = yaw_command_range[0] + (yaw_command_range[1] - yaw_command_range[0]) * ((-averaged_euler[1] - yaw_range[0]) / (yaw_range[1] - yaw_range[0]))
                command_array.data[20] = int(pitch)
                command_array.data[21] = int(roll)
                command_array.data[22] = int(yaw)
                publisher.publish(command_array)

        cv2.imshow("demo", frame)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()