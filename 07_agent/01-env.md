from onnxruntime import package_name

### 1. 使用 conda 创建虚拟环境并指定 python 版本

#### 安装 conda


#### 创建虚拟环境
```bash
# 创建虚拟环境，指定版本
conda create -n myenv python=3.12

```

#### 管理环境

```bash

```

### 2.查看 pip 安装的包版本

#### 查看自己安装的包

```bash
# 查看包的详细信息

pip show -v package_name

# 查看包安装的位置

pip show -f package_name
```

### 3. 使用 requirements.txt

```bash
# 生成当前环境的所有包列表
pip freesze > requirements.txt

# 只包含直接安装的包

pip list --format=freesze > requirements.txt

```
#### 使用 requirements.txt 
```bash


```