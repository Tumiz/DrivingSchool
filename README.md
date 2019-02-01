# DrivingSchool

DrivingSchool is a web-based 3D simulator for reinforcement learning algorithms.

## Requirements

* Python: >=3.0
* WebGL is supported on your web browser

## Preparation

1. Install the python web framework - tornado:

    pip install tornado

2. (optional) Install the deep learning platform - pytorch from https://pytorch.org/get-started/locally/

3. Download the codes of DrivingSchool:

    git clone https://github.com/Tumiz/DrivingSchool.git

## Run

1. Open a terminal in the folder of DrivingSchool and run the following command:

    python server.py

2. Open a web tab and go to:

    http://localhost:8080/

## Operations

|    Wanted | Key 
----------- |-----
 Create a car| Ctrl + Mouse Left Click 
Choose a car| Mouse Left Click on the car
Acceleration | W
Deceleration| S
Turn Left| A
Turn Right|D
Stop simulation | Click the Button Stop
Reset all to origin| Click the Button Reset
Reconnect to the server|Click the Button Start