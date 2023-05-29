# 删除所有tag为none的镜像
docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
# 删除tmp文件
find . -name '__pycache__' -type d -exec rm -rf {} \;