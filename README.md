# CS580-Final-Project

Vision-Guided Conveyor Sorting with a UR5e Arm

Project Team: Quincy Sorrentino (qsorrentino@wisc.edu), Yu-Hsin Lin (ylin459@wisc.edu), Pranit Vaikuntam (vaikuntam@wisc.edu)

**Project Setting:** The task we plan to complete for this project is sorting items on a conveyor belt using a robot arm and camera. With a similar setup used in Predictive Control of Cooperative Robots Sharing Common Workspace but with only one arm. As a baseline we will only sort objects into two classes but can expand as time and scope permits. Two classes is enough to prove the basic mechanics of our system, and additional classes add little technical value. For environment and manipulator we will use the Webots ConveyorBelt PROTO, the ure sample world, an overhead camera for perception and the UR5e robot since the forward kinematics are already defined. This task is particularly interesting because of its applicability across various industries, such as a warehouse where packages need sorting, or in the food industry where food with defects must be removed from the conveyor as part of quality checks.

**Techniques:** We will be using Jacobian-based inverse kinematics to manipulate the robot arm. For perception we have intentionally left this vague as it is the most flexible part of the project. Two options we will choose from are using a pretrained vision model such as YOLO or a more traditional vision technique such as Canny edge detection and color segmentation on simple blocks. We will also need to calculate the transforms between the detections and the robot's base frame. We will be using a Kalman filter to estimate the state of the tracked object as it moves along the conveyor belt. For task planning we will be using two different algorithms and making the decision on which one to keep based on our experiments. The first is a greedy picking algorithm that chooses the object closest to the end effector to pick and sort while the other is a deadline aware algorithm that will use the estimated trajectory to pick the object that is closest to leaving the workspace.

**Evaluation:** 
**Experiment 1:** Object Tracking Under Visual Uncertainty: We will introduce different levels of noise and lighting conditions into the camera detections. We will measure object-position estimation error, successful detection rate, and overall sorting accuracy. This experiment will evaluate which environmental conditions the perception task is viable in. 

## Vision setup

The overhead camera controller (`controllers/overhead_vision`) runs a pretrained YOLO11n model and needs a Python venv with ultralytics.

1. From the project root, create the venv:
   ```
   python3 -m venv .venv
   .venv/bin/pip install ultralytics opencv-python numpy
   ```
2. Copy `controllers/overhead_vision/runtime.ini.example` to `controllers/overhead_vision/runtime.ini` and set `COMMAND` to the **absolute** path of your `.venv/bin/python` (Webots does not resolve relative paths). `runtime.ini` is git-ignored because the path is different on each machine.
3. Open `world/conveyor_ur5e.wbt` and press Play. The console should print the ultralytics version, then one line per detection. The YOLO weights (`yolo11n.pt`) download automatically on the first run.

Detections are sent over the Emitter on channel 1 as JSON, one message per object: `{"t": sim_time, "class": "bottle" | "clock" | "cat", "conf": ..., "x": ..., "y": ...}` (world coordinates, meters).

