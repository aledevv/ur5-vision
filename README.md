<p align='center'>
    <h1 align="center">VISION NODE </h1>  
</p>

This is the vision node of the [robotics ur5 project](https://github.com/aledevv/robotics_project_ur5).

## INSTALLATION
The project has been developed and tested on Ubuntu 20.04 with ROS Noetic, also we used the [locosim](https://github.com/mfocchi/locosim) repository for the ur5 simulation. The installation of the project is the following:
1) Clone the [locosim repository](https://github.com/mfocchi/locosim) and follow the respective instructions to install it
2) Clone this [repository](https://github.com/aledevv/robotics_project_ur5) in the `ros_ws/src` folder of the catkin workspace
3) Install the vision dependencies with the following command:
- Install YOLOv5 dependencies
   
```BASH
cd ~
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip3 install -r requirements.txt
```
- Intall the other dependencies
```BASH
pip install torchvision==0.13.0
```
4) Compile the project with the following command:
```BASH
cd ~/ros_ws
catkin_make install
source install/setup.bash
```

## HOW TO RUN THE MODULE
### SETUP (for the entire project as well)
Inside  ``~/ros_ws/src/locosim/robot_control/base_controllers/params.py`` go to the line 46 and set:
```
'gripper_sim': True,
```
Now we have to add our .word file
Go to  ``~/ros_ws/src/robotic_project``

Then we copy the models inside the worlds directory :
```
cd ~/ros_ws/src/robotics_project_ur5
cp -r Models ~/ros_ws/src/locosim/ros_impedance_controller/worlds
```
Now add the world.world file
```BASH
cp world/world.world ~/ros_ws/src/locosim/ros_impedance_controller/worlds
```
Last thing is to modify the ur5_generic.py file in the locosim project adding the following line at line 71
```PYTHON
self.world_name = 'world.world'
```
Now we are able to run the project.

### RUN VISION NODE
For running the project you need to run the following commands:
1) Run in one window the locosim simulation with the following command:
```BASH
python3 ~/ros_ws/src/locosim/robot_control/base_controllers/ur5_generic.py
```
2) Run in another window the vision node with the following command:
```BASH
. ~/ros_ws/devel/setup.bash
roscd vision
cd scripts
python3 vision-node.py
```
