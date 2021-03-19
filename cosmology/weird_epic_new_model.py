# this file contains the definitions of many models, each of which resemble one try
# we will not go into far detail how each model works, we will give a short description
# each motivated by a finding or an idea

# as a simple benchmark we have the score one would achieve when just guessing the average convergence
# this value is: 0.00065

import tensorflow as tf

from tensorflow.python.ops import init_ops



import tensorflow as tf
import time
from performer import performer
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
from cosmology.model import make_model
from tensorflow.python.ops import init_ops

class HeNormal(init_ops.VarianceScaling):

  def __init__(self, scale=.002, seed=None):
    super(HeNormal, self).__init__(
        scale=scale, mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim, kernel_initializer=HeNormal())
        self.key_dense = Dense(embed_dim, kernel_initializer=HeNormal())
        self.value_dense = Dense(embed_dim, kernel_initializer=HeNormal())
        self.combine_heads = Dense(embed_dim, kernel_initializer=HeNormal())

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
        # return performer.fast_attention(query, key, value, self.projection_dim, self.embed_dim), None

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=tf.keras.activations.gelu, kernel_initializer=HeNormal()),
                Dropout(dropout),
                Dense(embed_dim, kernel_initializer=HeNormal()),
                Dropout(dropout),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class EpicNewModel(tf.keras.Model):
    def __init__(
        self,
        input_len,
        d_big,
        d_small,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        dropout=0.1,
    ):
        super(EpicNewModel, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.d_big = d_big

        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, d_big**2 + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))

        self.patch_proj = make_model(input_len, d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        # self.enc_layers = [
        #     performer.fast_attention([128, num_heads, mlp_dim], d_model, num_heads, mlp_dim, dropout)
        #     for _ in range(num_layers)
        # ]
        self.mlp_head = tf.keras.Sequential(
            [
                LayerNormalization(epsilon=1e-6),
                Dense(mlp_dim, activation=tf.keras.activations.gelu, kernel_initializer=HeNormal()), # moet gelu zijn, zelf custom toevoegen
                Dropout(dropout),
                Dense(num_classes, kernel_initializer=HeNormal()),
            ]
        )


    def call(self, inputs, training):
        batch_size = tf.shape(inputs)[0]
        shape = inputs.shape
        x = self.patch_proj(tf.reshape(inputs, (shape[0] * shape[1],) + shape[2:]))
        x = tf.reshape(x, (shape[0], self.d_big**2, self.d_model))

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x




    def get_config(self):
        return {'seed': self.seed}




def make_epic_model(d_big, d_small, depth):
    """
    This model uses a new architecture, called a transformer.
    Transformers are the most general neural network architecture
    that currently exists, but require too much compute and achieved
    relatively low performance on our training setup.

    Result: 0.00058
    """
    model = EpicNewModel(
        input_len=24,
        d_big=d_big,
        d_small=d_small,
        num_layers=4,
        num_classes=depth,
        d_model=128,
        num_heads=8,
        mlp_dim=128,
        dropout=0.1,

    )
    return model


