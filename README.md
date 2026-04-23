# Monoped RL Hopping

## Building the workspace (first time only)
cd /root/monoped_ws
catkin_make_isolated --install
source devel_isolated/setup.sh

## Docker Setup
docker run -it --rm \
  --name monoped_session \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/monoped_ws:/root/monoped_ws \
  ntklab/monoped_rl:latest bash

## Inside container
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh

## Terminal 1 — Launch Gazebo
roslaunch my_legged_robots_sims main.launch

## Terminal 2 — Start Training
cd src/my_hopper_training/src
roslaunch my_hopper_training main.launch

## Terminal 2 — Running Inference
Copy the trained model into the container first:
docker cp models_export/monoped_JUMP_d.zip monoped_session:/root/monoped_ws/models/

Then inside the container:
cd src/my_hopper_training/src
python3 inference.py
