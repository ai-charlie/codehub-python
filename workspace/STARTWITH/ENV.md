# 个人开发环境配置

- docker内cuda和本地的cuda版本不用一致，只依赖于驱动版本，139是3090的显卡，支持515.76的
- tensorrt 支持10.2 和 11.系列

## 使用脚本运行开发环境
sh run.sh
```bash
docker run --gpus all \
-v ${PWD}:/workspace \
--shm-size=8gb \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-p 18080:18080 \
-it tensorrt:trt8.5.1-cu11.3-ubuntu2004
```

## 使用flake8 配置代码格式化

### 下载 flake8 插件
### pip install yapf
### 修改vscode配置
```json
    "python.formatting.provider": "yapf",
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
```
### 配置项目 .flake8 文件
```flake8
[flake8]
max-line-length = 80
; Black violates PEP8
based_on_style = pep8
select = C,E,F,W,B
ignore = F401, W291, E501, W293, F541, F841, E126, W504, E402
spaces_before_comment = 4
split_before_logical_operator = true
exclude=.git, .pytest_cache, .tox, .vscode, venv, .venv, .idea, __pycache__, datasets, data
```