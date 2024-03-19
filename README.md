# moma_grasp_ws

## launch docker
```bash
cd moma_grasp_ws/src/moma/docker/
sudo ./run_docker.sh -d ghcr.io/ethz-asl/moma:demo -n moma_demo -w ~/moma_grasp_ws
```

## launch demo
in one terminal
```bash
roslaunch moma_gazebo panda_grasp.launch
```
in second terminal
first access to the running container
```bash
sudo docker exec -it moma_demo bash
```
then launch the grasping demo
```bash
roslaunch grasp_demo grasp_demo.launch launch_rviz:=true
```