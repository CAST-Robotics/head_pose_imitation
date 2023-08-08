#!/usr/bin/python3

import argparse
import service
import cv2
import roslib
import rospy
from std_msgs.msg import Int32MultiArray
from collections import deque
import numpy as np


def main(color=(224, 255, 255)):

    config_path = roslib.packages.get_pkg_dir('head_pose_imitation') + "/config"
    print(config_path)

    fd = service.UltraLightFaceDetecion(config_path + "/weights/RFB-320.tflite",
                                       conf_threshold=0.95)

    fa = service.DepthFacialLandmarks(config_path + "/weights/sparse_face.tflite")

    handler = getattr(service, "pose")
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

        for results in fa.get_landmarks(feed, boxes):
            euler = handler(frame, results, color)
            euler_queue.append(euler)  # Add euler angles to the moving average queue

        if len(euler_queue) > 0:
            averaged_euler = np.mean(euler_queue, axis=0)  # Calculate the average of euler angles
            print(f"Pitch: {averaged_euler[0]}; Yaw: {averaged_euler[1]}; Roll: {averaged_euler[2]};")
            pitch = 110 + (180 - 110) * ((averaged_euler[0] - (-30)) / (60 - (-30)))
            roll = 100 + (190 - 100) * ((-averaged_euler[2] - (-50)) / (50 - (-50)))
            yaw = 90 + (210 - 90) * ((-averaged_euler[1] - (-90)) / (90 - (-90)))
            command_array.data[20] = int(pitch)
            command_array.data[21] = int(roll)
            command_array.data[22] = int(yaw)
            publisher.publish(command_array)

        cv2.imshow("demo", frame)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()