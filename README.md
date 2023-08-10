# Head Pose Imitation
<p align="center">
<img src="media/imitation.gif"/>
</p>

This is a ROS package designed for head pose imitation on the Surena V humanoid robot. 

To estimate human head pose, we utilized the ["Dense Head Pose Estimation"](https://github.com/1996scarlet/Dense-Head-Pose-Estimation) project. The obtained head pose information is then mapped onto our robot for imitation purposes.

## Configuration

- The configuration files for the face detection and facial landmarks models should be placed in the `config/weights` directory.

- You can adjust the thresholds and parameters for face detection and facial landmarks in the `main.py` file.

- The mapping from detected angles space to motor commands space can be modified by updating the constants in the `main.py` file.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## To do

- Adding simulation support