import os
import tensorflow as tf
from tensorflow.contrib.training import HParams
import layers
import ops
import numpy as np

def encoder(self, features, seq_length, dim, dropo, is_training):
    with tf.variable_scope('encoder'):
      
      encoded = tf.identity(features, 'features')
      seq_length = tf.identity(seq_length, 'input_seq_length')
      
      hp = HParams(
          num_layers=3,
          num_head=8,
          dim_head=64,
          dim_head_v=64,
          
          ffnn_sublayer=True,
          ff_dim=2048,
          self_attention_sublayer_residual_and_norm=True,

           pi=3.14,
           pos_t=680,
           pos_abs=680,
           pos_m1=8,
           pos_m2=4,
           dim_p=6,
           batch_size=16)

      def inv_pos(sequence_len):
          pos_ind = tf.range(sequence_len - 1, -1, -1.0)
        
          def rpr_pos_out(num_t, p_ind):
              inv_freq = (2.0 * hp.pi * p_ind) / (num_t)
              p_cos = tf.expand_dims(tf.cos(inv_freq), 1)
              p_sin = tf.expand_dims(tf.sin(inv_freq), 1)
              pos = tf.concat([p_cos, p_sin], -1)
              return pos
         
          pos1 = rpr_pos_out(hp.pos_t, pos_ind)
          pos2 = rpr_pos_out(hp.pos_m1, pos_ind)
          pos3 = rpr_pos_out(hp.pos_m2, pos_ind)
        
          _pos = tf.concat([pos1, pos2, pos3], -1)
          _pos = tf.tile(tf.expand_dims(_pos, 0), [hp.batch_size, 1, 1])
        
          return _pos
      
      def dir_pos(sequence_len):
          pos_ind = tf.range(0.0, -sequence_len,-1)
        
          def rpr_pos_out(num_t, p_ind):
              inv_freq = (2.0 * hp.pi * p_ind) / (num_t)
              p_cos = tf.expand_dims(tf.cos(inv_freq), 1)
              p_sin = tf.expand_dims(tf.sin(inv_freq), 1)
              pos = tf.concat([p_cos, p_sin], -1)
              return pos
          
          pos1 = rpr_pos_out(hp.pos_t, pos_ind)
          pos2 = rpr_pos_out(hp.pos_m1, pos_ind)
          pos3 = rpr_pos_out(hp.pos_m2, pos_ind)
          
          _pos = tf.concat([pos1, pos2, pos3], -1)
          _pos = tf.tile(tf.expand_dims(_pos, 0), [hp.batch_size, 1, 1])
          
          return _pos
       
      def positional_encoding_6d(sentence_length):
          pos_ind = tf.range(sentence_length)
          def pos_out(num_t, p_ind):
              p_enc = np.array([(2 * hp.pi * np.float32(t)) / np.float32(num_t) for t in range(hp.pos_abs)])
              p_cos = tf.convert_to_tensor(np.cos(p_enc), tf.float32)
              p_sin = tf.convert_to_tensor(np.sin(p_enc), tf.float32)
              
              _p_cos = tf.expand_dims(tf.nn.embedding_lookup(p_cos, p_ind), 1)
              _p_sin = tf.expand_dims(tf.nn.embedding_lookup(p_sin, p_ind), 1)
              
              pos = tf.concat([_p_cos, _p_sin], -1)
              return pos
            
          pos1 = pos_out(hp.pos_abs, pos_ind)
          pos2 = pos_out(hp.pos_m1, pos_ind)
          pos3 = pos_out(hp.pos_m2, pos_ind)
          _pos = tf.concat([pos1, pos2, pos3], -1)
          _pos = tf.tile(tf.expand_dims(_pos, 0), [hp.batch_size, 1, 1])
            
          return _pos

      def mask_for_rpr(qlen):
          rpr_mask = tf.ones([qlen, qlen])
          rpr_mask_u = tf.matrix_band_part(rpr_mask, 0, -1)
          rpr_mask_dia = tf.matrix_band_part(rpr_mask, 0, 0)
          rpr_ret = rpr_mask_u - rpr_mask_dia
        
          return rpr_ret
    
      def _create_mask(qlen, sel_time=25, mask_before=True, mask_after=True):
          attn_mask = tf.ones([qlen, qlen])
          mask_u = tf.matrix_band_part(attn_mask, 0, -1)
          if mask_before:
             mask_l = tf.matrix_band_part(attn_mask, -1, 0)
             mask_sel = tf.matrix_band_part(attn_mask, sel_time, 0)
             mask_lower = mask_l - mask_sel
          else:
             mask_lower = tf.zeros([qlen, qlen])
        
          if mask_after:
             mask_up = tf.matrix_band_part(attn_mask, 0, sel_time)
             ret = mask_u - mask_up + mask_lower
          else:
             mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
             ret = mask_u - mask_dia + mask_lower
          return ret

      def rel_shift_l(x, qlen, bsz, n_head):
          x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
          x = tf.reshape(x, [bsz, n_head, qlen + 1, qlen])
          x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
          x = tf.reshape(x, [bsz, n_head, qlen, qlen])
          return x
      
      def rel_shift_r(x, qlen, bsz, n_head):
          x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, 1]])
          x = tf.reshape(x, [bsz, n_head, qlen + 1, qlen])
          x = tf.slice(x, [0, 0, 0, 0], [-1, -1, qlen, -1])
          x = tf.reshape(x, [bsz, n_head, qlen, qlen])
          return x

      def multi_head_encoder_layer(input_sequence, pos_l, pos_r, v_r_r, v_r_l, mask_zeropad, mask_time, dim, max_t, dropo, is_training, num_l):
          E_c = input_sequence.get_shape().as_list()[-1]
        
          Q = tf.layers.dense(input_sequence, hp.num_head * hp.dim_head, activation=None, use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              name='Q')
          K = tf.layers.dense(input_sequence, hp.num_head * hp.dim_head, activation=None, use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              name='K')
          V = tf.layers.dense(input_sequence, hp.num_head * hp.dim_head_v, activation=None, use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              name='V')
         
          # B T d
          P_k_r = tf.layers.dense(pos_r, hp.num_head * hp.dim_p, activation=None, use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='P_k_r')
          P_k_l = tf.layers.dense(pos_l, hp.num_head * hp.dim_p, activation=None, use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='P_k_l')

          # B T H D
          Qs = tf.reshape(Q, [hp.batch_size, max_t, hp.num_head, hp.dim_head])
          Ks = tf.reshape(K, [hp.batch_size, max_t, hp.num_head, hp.dim_head])
          Vs = tf.reshape(V, [hp.batch_size, max_t, hp.num_head, hp.dim_head_v])
          # B H T D
          Qs = tf.transpose(Qs, [0, 2, 1, 3])
          Ks = tf.transpose(Ks, [0, 2, 1, 3])
          Vs = tf.transpose(Vs, [0, 2, 1, 3])
        
          # B T H d
          P_ks_l = tf.reshape(P_k_l, [hp.batch_size, max_t, hp.num_head, hp.dim_p])
          # B H T d
          P_ks_l = tf.transpose(P_ks_l, [0, 2, 1, 3])
          # B T H d
          P_ks_r = tf.reshape(P_k_r, [hp.batch_size, max_t, hp.num_head, hp.dim_p])
          # B H T d
          P_ks_r = tf.transpose(P_ks_r, [0, 2, 1, 3])

          # B H T D * B H D T => B H T T
          attn_c = tf.matmul(Qs, Ks, transpose_b=True)
          attn_c = (attn_c / (hp.dim_head ** 0.5))

          # B H T d * B H d T => B H T T
          attn_p_l = tf.einsum('hid,bhjd->bhij', v_r_l, P_ks_l)
          attn_p_l = rel_shift_l(attn_p_l, max_t, hp.batch_size, hp.num_head)
          attn_p_l = tf.matrix_band_part(attn_p_l, -1, 0)

          attn_p_r = tf.einsum('hid,bhjd->bhij', v_r_r, P_ks_r)
          attn_p_r = rel_shift_r(attn_p_r, max_t, hp.batch_size, hp.num_head)
          attn_p_r = tf.matrix_band_part(attn_p_r, 0, -1) - tf.matrix_band_part(attn_p_r, 0, 0)

          attn_p = attn_p_r + attn_p_l
          attn_p = attn_p / (hp.dim_p ** 0.5)
        
          # B H T T
          logits = attn_c + attn_p
          # B T
          mask_zeropad = tf.to_float(mask_zeropad)
          mask_zeropad = mask_zeropad[:, tf.newaxis, tf.newaxis, :]
          logits += (mask_zeropad * (-1e30))
          # T T => None None T T
          mask_time = mask_time[tf.newaxis, tf.newaxis, :, :]
          logits = logits * (1 - mask_time) - 1e30 * mask_time
          # last dimension softmax
          attn_weight = tf.nn.softmax(logits)
          attn_weight = tf.layers.dropout(attn_weight, dropo, training=is_training)
          ######################################
          # B H T T * B H T D => B H T D Transpose
          outputs = tf.matmul(attn_weight, Vs)
          # B H T D => B T H D Reshape (concate)
          outputs = tf.transpose(outputs, [0, 2, 1, 3])
          # B T H D => B T D
          outputs = tf.reshape(outputs, [hp.batch_size, max_t, hp.num_head * hp.dim_head_v])
          # Linear Transform
          outputs = tf.layers.dense(outputs, E_c, activation=None, use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='attan_output')
          outputs = tf.layers.dropout(outputs, dropo, training=is_training)

          if hp.self_attention_sublayer_residual_and_norm:
             self_attention_layer = tf.add(outputs, input_sequence)
             self_attention_layer = tf.contrib.layers.layer_norm(self_attention_layer, begin_norm_axis=-1)

          if hp.ffnn_sublayer:
             ffnn_sublayer_output = tf.layers.dense(self_attention_layer, hp.ff_dim, activation=tf.nn.relu,
                                                    use_bias=True,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='ff_1')
             ffnn_sublayer_output = tf.layers.dropout(ffnn_sublayer_output, dropo, training=is_training)

             ffnn_sublayer_output = tf.layers.dense(ffnn_sublayer_output, E_c, activation=None,
                                                    use_bias=True,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='ff_2')
             ffnn_sublayer_output = tf.layers.dropout(ffnn_sublayer_output, dropo, training=is_training)

             ffnn_sublayer_output = tf.add(ffnn_sublayer_output, self_attention_layer)
             encoder_layer_output = tf.contrib.layers.layer_norm(ffnn_sublayer_output, begin_norm_axis=-1)

          else:
             encoder_layer_output = self_attention_layer

          return encoder_layer_output

      def transformerClassifier(emb, mask_zeropad, mask_time, dim, max_sentence_length, dropo, is_training):
          # H 1 d
          v_r_l = tf.tile(tf.get_variable('v_r_l', [hp.num_head, 1, hp.dim_p],
                                          initializer=tf.contrib.layers.xavier_initializer()),
                          [1, max_sentence_length, 1])
          v_r_r = tf.tile(tf.get_variable('v_r_r', [hp.num_head, 1, hp.dim_p],
                                          initializer=tf.contrib.layers.xavier_initializer()),
                          [1, max_sentence_length, 1])

          pos_l = inv_pos(max_sentence_length)
          pos_r = dir_pos(max_sentence_length)


          emb = tf.layers.dropout(emb, dropo, training=is_training)
          pos_l = tf.layers.dropout(pos_l, dropo, training=is_training)
          pos_r = tf.layers.dropout(pos_r, dropo, training=is_training)

          for i in range(1, hp.num_layers + 1):
               #with tf.variable_scope("Stack-Layer-{0}".format(i)):
               with tf.variable_scope("Share-Layer"):
                    encoder_output = multi_head_encoder_layer(emb, pos_l, pos_r, v_r_r, v_r_l,
                                                              mask_zeropad, mask_time, 
                                                              dim, max_sentence_length, dropo, is_training,num_l=i)
                  
                    emb = tf.layers.dropout(encoder_output, dropo, training=is_training)

                return encoder_output

      with tf.variable_scope('Transformer', reuse=tf.AUTO_REUSE):
           max_time = tf.shape(encoded)[1]
           dim_enc = encoded.shape.as_list()[-1]

           with tf.variable_scope('CNN'):
                encoded = tf.reshape(encoded, [hp.batch_size, max_time, dim_enc, 1])
                w_conv1 = tf.get_variable('w_conv1', shape=[3, 3, 1, 1],
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
                w_conv2 = tf.get_variable('w_conv2', shape=[3, 3, 1, 1],
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
                encoded_conv1 = tf.nn.conv2d(encoded, w_conv1,
                                             strides=[1, 2, 1, 1],
                                             padding='SAME')
                encoded_conv2 = tf.nn.conv2d(encoded_conv1, w_conv2,
                                             strides=[1, 2, 1, 1],
                                             padding='SAME')
                encoded = tf.reshape(encoded_conv2, [hp.batch_size, -1, dim_enc])

                seq_length = tf.to_int32(tf.ceil(
                    tf.to_float(seq_length) /
                    float(self.conf['subsample'])))
                # T/2
                max_sentence_length = tf.shape(encoded)[1]

                masks_zeropad = tf.equal(encoded, 0)[:, :, 0]
                masks_time = _create_mask(max_sentence_length, 25, True, True)
                outputs = transformerClassifier(encoded, masks_zeropad, masks_time, dim, max_sentence_length, dropo, is_training)
                abs_pos = positional_encoding_6d(max_sentence_length)
                encode_out = tf.concat([outputs, abs_pos], -1)

            encoded = tf.identity(encode_out, 'encoded')
            seq_length = tf.identity(seq_length, 'output_seq_length')
            num_head = hp.num_head

    return encoded, seq_length
