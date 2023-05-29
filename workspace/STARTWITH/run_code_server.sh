docker run --rm -it --gpus all --name code  -p 10367:8080 \
  -v "$HOME:/home/coder/" \
  -u "$(id -u):$(id -g)" \
  -e "DOCKER_USER=coder" \
  -d  codercom/code-server:latest
