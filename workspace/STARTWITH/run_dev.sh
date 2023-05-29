docker run --gpus all \
    -p 10367 \
    -v ${PWD}:/workspace \
    -v /etc/localtime:/etc/localtime \
    --name zhanglq \
    --shm-size=64gb \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -it one:tf2.9-torch1.10-trt8.5.1-cu11.3-2004
