# Monoped RL Hopping

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

## Terminal 2 — Run Inference
cd src/my_hopper_training/src
python3 inference.py
