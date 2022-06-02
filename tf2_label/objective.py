# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Contrastive loss functions."""

from absl import flags

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

LARGE_NUM = 1e9


def add_supervised_loss(labels, logits):
    """Compute mean supervised loss over local batch."""
    losses = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                    logits)
    return tf.reduce_mean(losses)


def add_contrastive_loss(hidden,
                         labels,
                         hidden_norm=True,
                         temperature=1.0,
                         strategy=None):
    """Compute loss for model.

    Args:
      hidden: hidden vector (`Tensor`) of shape (bsz, dim).
      hidden_norm: whether or not to use normalization on the hidden vector.
      temperature: a `floating` number for temperature scaling.
      strategy: context information for tpu.

    Returns:
      A loss scalar.
      The logits for contrastive prediction task.
      The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    '''
    # trasform label to mat
    labels = tf.argmax(labels['labels'], axis=1) + 1  # shape [batch_size,]
    labels = tf.cast(labels, tf.float32)
    y_h = tf.expand_dims(labels, axis=0)
    y_v = tf.expand_dims(labels, axis=1)
    y_mat_h = tf.concat([y_h for i in range(y_h.shape[1])], axis=0)
    y_mat_v = tf.concat([y_v for i in range(y_v.shape[0])], axis=1)
    y_mat = y_mat_h - y_mat_v
    y_abs = y_mat * tf.transpose(y_mat)
    mat_label = tf.exp(y_abs * LARGE_NUM)
    '''
    # Gather hidden1/hidden2 across replicas and create local labels.
    if strategy is not None:
        hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
        hidden2_large = tpu_cross_replica_concat(hidden2, strategy)

        # TODO: use mask to update all batches similar label as current one
        label = tf.argmax(labels['labels'], axis=1) + 1
        y_large = tpu_cross_replica_concat(label, strategy)
        enlarged_batch_size = tf.shape(hidden1_large)[0]
        label = tf.cast(label, tf.float32)
        y_large = tf.cast(y_large, tf.float32)
        y_h_large = tf.expand_dims(y_large, axis=0)
        label_v = tf.expand_dims(label, axis=1)
        y_mat_h = tf.concat([y_h_large for i in range(label.shape[0])], axis=0)
        y_mat = y_mat_h-label_v

        mat_label_large = tf.exp(tf.negative(tf.abs(y_mat)) * LARGE_NUM)
        masks = mat_label_large
        # TODO: change multigpu label to pos
        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.cast(
            tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        self_labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
        self_masks = tf.one_hot(labels_idx, enlarged_batch_size)
        '''
        left_mask = tf.zeros([batch_size,replica_id * batch_size],tf.float32)
        right_mask_pos = enlarged_batch_size - (replica_id+1) * batch_size
        right_mask = tf.zeros([batch_size, right_mask_pos], tf.float32)
        masks = tf.concat([left_mask, mat_label], axis=1)
        masks = tf.concat([masks, right_mask], axis=1)
        '''
        #masks = tf.one_hot(labels_idx, enlarged_batch_size)
    else:
        hidden1_large = hidden1
        hidden2_large = hidden2
        self_labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        # trasform label to mat
        label = tf.argmax(labels['labels'], axis=1) + 1  # shape [batch_size,]
        label = tf.cast(label, tf.float32)
        y_h = tf.expand_dims(label, axis=0)
        y_v = tf.expand_dims(label, axis=1)
        y_mat_h = tf.concat([y_h for i in range(y_h.shape[1])], axis=0)
        #y_mat_v = tf.concat([y_v for i in range(y_v.shape[0])], axis=1)
        y_mat = y_mat_h - y_v
        y_abs = y_mat * tf.transpose(y_mat)
        mat_label = tf.exp(y_abs * LARGE_NUM)
        masks = mat_label
        self_masks = tf.one_hot(tf.range(batch_size), batch_size)
        #lebal_left = tf.nn.softmax(masks)
        #lebal_right = tf.zeros_like(masks)
        #self_labels = tf.concat([lebal_left, lebal_right], axis=1)
    #pos_mask = masks - self_masks
    #mul_buff = -pos_mask*2 + 1

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    #logits_ab = logits_ab*mul_buff + pos_mask # * LARGE_NUM
    logits_ab = logits_ab - masks * LARGE_NUM
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
    #logits_ba = logits_ba*mul_buff + pos_mask # * LARGE_NUM
    logits_ba = logits_ba - masks * LARGE_NUM

    # sim_a = tf.matmul(two_batch_label, tf.concat([logits_ab, logits_aa], axis=1), transpose_b=True) / temperature
    # sim_b = tf.matmul(two_batch_label, tf.concat([logits_ba, logits_bb], axis=1), transpose_b=True) / temperature

    sim_a = tf.concat([logits_ab, logits_aa], axis=1)
    sim_b = tf.concat([logits_ba, logits_bb], axis=1)

    # only positive labels
    # sim_a = tf.matmul(mat_label, logits_ab, transpose_b=True) / temperature
    # sim_b = tf.matmul(mat_label, logits_ba, transpose_b=True) / temperature

    # TODO: use teacher model loss for Knowledge Distillation
    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        self_labels, sim_a)
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        self_labels, sim_b)
    loss = tf.reduce_mean(loss_a + loss_b)

    return loss, logits_ab, self_labels


def tpu_cross_replica_concat(tensor, strategy=None):
    """Reduce a concatenation of the `tensor` across TPU cores.

    Args:
      tensor: tensor to concatenate.
      strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.

    Returns:
      Tensor of the same rank as `tensor` with first dimension `num_replicas`
      times larger.
    """
    if strategy is None or strategy.num_replicas_in_sync <= 1:
        return tensor

    num_replicas = strategy.num_replicas_in_sync

    replica_context = tf.distribute.get_replica_context()
    with tf.name_scope('tpu_cross_replica_concat'):
        # This creates a tensor that is like the input tensor but has an added
        # replica dimension as the outermost dimension. On each replica it will
        # contain the local values and zeros for all other values that need to be
        # fetched from other replicas.
        ext_tensor = tf.scatter_nd(
            indices=[[replica_context.replica_id_in_sync_group]],
            updates=[tensor],
            shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0))

        # As every value is only present on one replica and 0 in all others, adding
        # them all together will result in the full tensor on all replicas.
        ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                                ext_tensor)

        # Flatten the replica dimension.
        # The first dimension size will be: tensor.shape[0] * num_replicas
        # Using [-1] trick to support also scalar input.
        return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
