import os
# 在导入 TensorFlow 之前设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 最高级别屏蔽
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 禁用 GPU 日志
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("hello world!!!!")