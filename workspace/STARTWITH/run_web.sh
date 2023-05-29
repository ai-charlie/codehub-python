docker run --gpus all \
    -v ${PWD}:/workspace \
    --shm-size=8gb \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 3001:3001 \
    -p 3000:3000 \
    -it web-front:latest
