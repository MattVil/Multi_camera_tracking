#!/bin/bash

echo "args:"
echo "   -i: image name (see 'docker images')"
echo "   -d: project directory (absolute path)"
echo "   -n: container name"

while getopts ":i:d:n:" opt; do
  case $opt in
    i) IMAGE_NAME="$OPTARG"
    ;;
    d) PROJECT_DIR="$OPTARG"
    ;;
    n) CONTAINER_NAME="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG. -i image_name -d shared_directory -n container_name" >&2
    ;;
  esac
done

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth-n
xauth nlist :0 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run -p 0.0.0.0:6006:6006 \
	--cap-add=SYS_PTRACE \
	--shm-size=1g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--security-opt \
	seccomp=unconfined \
	--gpus all \
	--name ${CONTAINER_NAME} \
	-e "TERM=xterm-256color" \
	--env="QT_X11_NO_MITSHM=1" \
	-v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH \
	-v ${PROJECT_DIR}:/workspace \
	-d ${IMAGE_NAME} \
	tail -f /dev/null
