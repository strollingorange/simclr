import tensorflow as tf
import model as model_lib

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  model = model_lib.Model(10)

checkpoint_manager2 = tf.train.CheckpointManager(
        tf.train.Checkpoint(model=model),
        directory="./tmp/simclr_test_ft",
        max_to_keep=5)

checkpoint_manager2.checkpoint.restore("./tmp/simclr_test").expect_partial()
#print(checkpoint_manager2.checkpoint)