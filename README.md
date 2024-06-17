# moma_grasp_ws
In order to reproduce the results follow the following steps:

## Prequisites
Frist you should have docker installed on your machine.

## 1) Clone the repo
Clone the current repository
```bash
git clone <repo-link>
```

## 2) build moma docker image and build the workspace
On one terminal access the docker container
```bash
cd moma_grasp_ws/src/moma/docker/
sudo ./run_docker.sh -d ghcr.io/ethz-asl/moma:demo -n moma_demo -w ~/moma_grasp_ws
```
note that the flag -w specify the location where the folder is sotred on your machine.

Then lets build the workspace
```bash
catkin build
source devel/setup.bash
```

## 3) launch the evalutation
In order to launch the evaluation additional terminal, accessing the docker container, should be opened with the following command:
```bash 
sudo docker exec -it moma_demo bash
```

On the first terminal that you have arleady open launch the gazebo simulation:
```bash
roslaunch moma_gazebo panda_grasp.launch
```

On a second terminal launch the grasping demo:
```bash
roslaunch grasp_demo grasp_demo.launch 
```

On a third terminal run the evaluation node:
```bash
rosrun next_best_view next_best_view_node.py 
```

On a fourth terminal launch the next best view node:
```bash
rosrun evaluation evaluation_node.py
```

at this point the evaluation starts and data are writte in the data_plotting folder


