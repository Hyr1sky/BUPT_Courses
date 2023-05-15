import tensorflow as tf
import pprint # 使用pprint 提高打印的可读性

NewCheck =tf.train.NewCheckpointReader("model.ckpt")
print("debug_string:\n")
pprint.pprint(NewCheck.debug_string().decode("utf-8")) #类型是str