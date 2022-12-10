## · 环境安装

### 1.安装Anaconda软件：网址：https://www.anaconda.com/

### 2.在conda中安装下列包：

- 创建一个新的conda环境：

  ```
  conda create -n TF2.1 python=3.7
  ```

- 安装英伟达的SDK10.1版本

  ```
  conda intsall cudatoolkit=10.1
  ```

- 安装英伟达的深度学习软件包7.6版本

  ```
  conda insatll cudnn=7.6
  ```

- 安装Tensorflow，并且指定2.1版本

  ```
  pip install tensorflow==2.1
  ```

  #### **注意:**  如果GPU不支持，则只需要安装tensorflow即可。

### 3.测试安装环境是否成功

1. ##### 打开终端

   ```
   conda activate TF2.1
   ```

2. ##### 进入python

   ```
   python
   ```

3. ##### 查看 Tensorflow 版本

   ```python
   # 导入tensorflow包
   import tensorflow as tf
   # 查看版本
   te=f.__version__
   ## 如果输出：“2.1.0”  则证明安装成功
   ```

**注意：** 报错解决：[TypeError: Descriptors cannot not be created directly.](https://www.cnblogs.com/kevin-hou1991/p/16358915.html)

```python
# 该错误是由于protobuf版本过高引起的
 1.卸载原来版本：
	pip uninstall protobuf
2.安装新版本
    pip install protobuf==3.19.0
```

