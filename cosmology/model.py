# this file contains the definitions of many models, each of which resemble one try
# we will not go into far detail how each model works, we will give a short description
# each motivated by a finding or an idea

# as a simple benchmark we have the score one would achieve when just guessing the average convergence
# this value is: 0.00065

import tensorflow as tf

from tensorflow.python.ops import init_ops



import tensorflow as tf
from performer import performer
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
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


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        input_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, input_size + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))

        self.patch_proj = tf.keras.models.Sequential([tf.keras.layers.BatchNormalization(), Dense(d_model, kernel_initializer=HeNormal(.000002))])
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
        x = self.patch_proj(inputs)

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


# a 2D residual block in python
def residual_block_2D(X, f, filters, is_convolutional_block=False, s=1):
    X_shortcut = X

    # Als convolutional is, moet main path krimpen
    strides = (s, s) if is_convolutional_block else (1, 1)
    X = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=strides, padding='valid',
                               kernel_initializer=HeNormal())(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(f, f), strides=(1, 1), padding='same',
                               kernel_initializer=HeNormal())(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               kernel_initializer=HeNormal())(X)
    X = tf.keras.layers.BatchNormalization()(X)

    # Als convolutional is, moet shortcut krimpen
    if is_convolutional_block:
        connection = tf.keras.layers.Conv2D(filters[2], (1, 1), strides=(s, s), padding='valid',
                                            kernel_initializer=HeNormal())
        X_shortcut = connection(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


# a 3D residual block in python
def residual_block_3D(X, f, filters, is_convolutional_block=False, s=1):
    X_shortcut = X

    strides = (s, s, 1) if is_convolutional_block else (1, 1, 1)
    X = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1, 1, 1), strides=strides, padding='valid',
                               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv3D(filters=filters[1], kernel_size=(f, f, f), strides=(1, 1, 1), padding='same',
                               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv3D(filters=filters[2], kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
                               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization()(X)

    # Als convolutional is, moet shortcut krimpen
    if is_convolutional_block:
        X_shortcut = tf.keras.layers.Conv3D(filters[2], (1, 1, 1), strides=(s, s, 1), padding='valid',
                                            kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def make_model(r, depth):
    """
    The first try. In this implementation we go by the basic ResNet recipe,
    which is to go as deep as possible, whilst squishing the image more and more,
    whilst simultaneously increasing the amount of kernels.

    Result: this network was not able to significantly surpass the benchmark
    """

    X_conv = tf.keras.layers.Input((2 * r + 1, 2 * r + 1, depth, 1))

    X = tf.keras.layers.Conv3D(16, (3, 3, 3), strides=(2, 2, 1), padding='same',
                               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X_conv)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.AveragePooling3D(pool_size=(1, 1, 1))(X)

    X = residual_block_3D(X, f=3, filters=[16, 16, 64], s=1, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [16, 16, 64])
    X = residual_block_3D(X, f=3, filters=[16, 16, 64], s=2, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [16, 16, 64])

    X = residual_block_3D(X, f=3, filters=[16, 16, 64], s=2, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [16, 16, 64])
    X = residual_block_3D(X, 3, [16, 16, 64])
    X = residual_block_3D(X, f=3, filters=[16, 16, 64], s=1, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [16, 16, 64])

    X = residual_block_3D(X, f=3, filters=[16, 16, 64], s=2, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [16, 16, 64])
    X = residual_block_3D(X, 3, [16, 16, 64])
    X = residual_block_3D(X, 3, [16, 16, 64])

    X = residual_block_3D(X, f=3, filters=[32, 32, 128], s=2, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [32, 32, 128])
    X = residual_block_3D(X, 3, [32, 32, 128])
    X = residual_block_3D(X, 3, [32, 32, 128])

    X = residual_block_3D(X, f=3, filters=[256, 256, 1024], s=1, is_convolutional_block=True)
    X = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)

    X = tf.keras.layers.Dense(depth)(X)

    model = tf.keras.models.Model(inputs=[X_conv], outputs=X, name='SortOfResNet50')
    return model


def make_model(r, depth):
    """
    This is a very simple model using just one resnet block.
    Weirdly it achieved better results than the complicated model above.
    We expect that this is because the above model compresses the image.
    In normal image classification, where a pixel is doesn't much matter.
    This is called translation invariance. But in our dataset, it is very
    important to know what the position is of the mass, since distance is
    a big factor in the gravitational force of galaxies. Compression loses
    this information. That is why this simple network, that does not compress,
    performs better than its predecessor.

    Result: 0.00060
    """
    X_conv = tf.keras.layers.Input((2 * r + 1, 2 * r + 1, depth, 1))

    X = residual_block_3D(X_conv, f=3, filters=[16, 16, 16], is_convolutional_block=True, s=2)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)

    X = tf.keras.layers.Dense(depth)(X)

    model = tf.keras.models.Model(inputs=[X_conv], outputs=X, name='SortOfResNet50')
    return model


def make_model(input_len, depth):
    """
    This model uses a new architecture, called a transformer.
    Transformers are the most general neural network architecture
    that currently exists, but require too much compute and achieved
    relatively low performance on our training setup.

    Result: 0.00058
    """
    model = VisionTransformer(
        input_size=input_len,
        num_layers=8,
        num_classes=depth,
        d_model=256,
        num_heads=32,
        mlp_dim=256,
        dropout=0.1,

    )
    return model



    model = VisionTransformer(
        input_size=input_len,
        num_layers=2,
        num_classes=depth,
        d_model=16,
        num_heads=4,
        mlp_dim=32,
        dropout=0.1,

    )
    return model


def make_models(r_a, r_b, r_c, r_d, depth):
    """
    This model is more complicated. We still make use of distance being an important factor,
    but also that the exact distance matters less and less when getting further away from the origin.
    So what we do is: we select 4 patches. One with radius 40, one with radius 20, one with radius 10
    and one with radius 5. Each of these gets their own subnetwork, with increasingly little compression.
    The outputs of these subnetworks get concatenated together to form one big network.
    Furthermore, we increase the size of the network in every dimension.
    As one last change we go from 3D to 2D. 2D is much less computationally intensive,
    which allows us to train bigger models. We do run into the drawback of having less spatial information.
    2D networks do not fully use the spatial information along the z-axis.

    Result: 0.00045
    """
    X_conv_a = tf.keras.layers.Input((2 * r_a + 1, 2 * r_a + 1, depth))
    X = tf.keras.layers.BatchNormalization()(X_conv_a)


    filters = 128
    X = residual_block_2D(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    # X = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(X)
    X_a = tf.keras.layers.Flatten()(X)

    X_conv_b = tf.keras.layers.Input((2 * r_b + 1, 2 * r_b + 1, depth))
    X = tf.keras.layers.BatchNormalization()(X_conv_b)

    filters = 32
    X = residual_block_2D(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    # X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(X)
    X_b = tf.keras.layers.Flatten()(X)

    X_conv_c = tf.keras.layers.Input((2 * r_c + 1, 2 * r_c + 1, depth))
    X = tf.keras.layers.BatchNormalization()(X_conv_c)


    filters = 64
    # X = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))(X_conv_c)
    X = residual_block_2D(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    X = residual_block_2D(X, 3, [filters, filters, filters])
    # X = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))(X)
    X_c = tf.keras.layers.Flatten()(X)

    X_conv_d = tf.keras.layers.Input((2 * r_d + 1, 2 * r_d + 1, depth))
    X = tf.keras.layers.BatchNormalization()(X_conv_d)

    X = tf.keras.layers.Flatten()(X)

    d_neurons = 256

    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X_d = tf.keras.layers.Dense(d_neurons, "relu")(X)

    concat = tf.keras.layers.Concatenate()([X_a, X_b, X_c, X_d])  #
    X = tf.keras.layers.BatchNormalization()(concat)

    d_neurons = 256

    X = tf.keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Dense(depth)(X)

    model = tf.keras.models.Model(inputs=[X_conv_a, X_conv_b, X_conv_c, X_conv_d], outputs=X, name='SortOfResNet50')
    return model

def make_models(r_a, r_b, r_c, r_d, depth):
    """
    This model is more complicated. We still make use of distance being an important factor,
    but also that the exact distance matters less and less when getting further away from the origin.
    So what we do is: we select 4 patches. One with radius 40, one with radius 20, one with radius 10
    and one with radius 5. Each of these gets their own subnetwork, with increasingly little compression.
    The outputs of these subnetworks get concatenated together to form one big network.
    Furthermore, we increase the size of the network in every dimension.
    As one last change we go from 3D to 2D. 2D is much less computationally intensive,
    which allows us to train bigger models. We do run into the drawback of having less spatial information.
    2D networks do not fully use the spatial information along the z-axis.

    Result: 0.00045
    """

    X_conv_a = tf.keras.layers.Input((2 * r_a + 1, 2 * r_a + 1, depth))
    # X_a = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), padding="same")(X_conv_a)
    # X_a = tf.keras.layers.BatchNormalization()(X_a)

    X_conv_b = tf.keras.layers.Input((2 * r_b + 1, 2 * r_b + 1, depth))
    # X_b = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same")(X_conv_b)
    # X_b = tf.keras.layers.BatchNormalization()(X_b)


    X_conv_c = tf.keras.layers.Input((2 * r_c + 1, 2 * r_c + 1, depth))
    # X_c = tf.keras.layers.BatchNormalization()(X_conv_c)

    # X = tf.keras.layers.Concatenate()([X_a, X_b, X_c])



    filters = 64
    size = 5
    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(X_conv_a)
    # X = tf.keras.layers.Conv2D(4*filters, size, activation="relu", kernel_initializer=HeNormal())(X_conv_b)
    X = residual_block_2D(X, size, [filters, filters, 4*filters], s=1, is_convolutional_block=True)
    X = residual_block_2D(X, size, [filters, filters, 4*filters])
    X = residual_block_2D(X, size, [filters, filters, 4*filters])
    X = residual_block_2D(X, size, [filters, filters, 4*filters])
    X = residual_block_2D(X, size, [filters, filters, 4*filters], s=1, is_convolutional_block=False)
    X = residual_block_2D(X, size, [filters, filters, 4*filters])
    X = residual_block_2D(X, size, [filters, filters, 4*filters])
    X = residual_block_2D(X, size, [filters, filters, 4*filters])
    X = residual_block_2D(X, size, [filters, filters, 4*filters], s=1, is_convolutional_block=False)
    concat = tf.keras.layers.Flatten()(X)

    X_conv_d = tf.keras.layers.Input((2 * r_d + 1, 2 * r_d + 1, depth))

    # X = tf.keras.layers.Flatten()(X_conv_d)

    d_neurons = 512

    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X_d = tf.keras.layers.Dense(d_neurons, "relu")(X)

    # concat = tf.keras.layers.Concatenate()([X_concat, X_d])  #

    d_neurons = 512

    X = tf.keras.layers.Dense(d_neurons, "relu", kernel_initializer=HeNormal())(concat)
    X = tf.keras.layers.Dense(d_neurons, "relu", kernel_initializer=HeNormal())(X)
    X = tf.keras.layers.Dense(d_neurons, "relu", kernel_initializer=HeNormal())(X)

    output_layer = tf.keras.layers.Dense(depth, kernel_initializer=HeNormal())
    print(f"cuckoo {output_layer.dtype_policy}, {output_layer.dtype}")

    output_layer = output_layer(X)

    model = tf.keras.models.Model(inputs=X_conv_a, outputs=output_layer, name='SortOfResNet50')
    return model


def make_models(r_a, r_b, r_c, r_d, depth):
    """
    This model mimics the above model, only with 3D oriented kernels.
    It is therefore obligatory that it uses less parameters.

    Result:
    """
    X_conv_a = tf.keras.layers.Input((2 * r_a + 1, 2 * r_a + 1, depth, 1))

    filters = 16
    X = residual_block_3D(X_conv_a, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = tf.keras.layers.AveragePooling3D()(X)
    X_a = tf.keras.layers.Flatten()(X)

    X_conv_b = tf.keras.layers.Input((2 * r_b + 1, 2 * r_b + 1, depth, 1))

    filters = 32
    X = residual_block_3D(X_conv_b, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = tf.keras.layers.AveragePooling3D()(X)
    X_b = tf.keras.layers.Flatten()(X)

    X_conv_c = tf.keras.layers.Input((2 * r_c + 1, 2 * r_c + 1, depth, 1))

    filters = 32
    X = residual_block_3D(X_conv_c, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = residual_block_3D(X, 3, [filters, filters, filters])
    X = tf.keras.layers.AveragePooling3D()(X)
    X_c = tf.keras.layers.Flatten()(X)

    X_conv_d = tf.keras.layers.Input((2 * r_d + 1, 2 * r_d + 1, depth, 1))
    X = tf.keras.layers.Flatten()(X_conv_d)

    d_neurons = 512

    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X_d = tf.keras.layers.Dense(d_neurons, "relu")(X)

    concat = tf.keras.layers.Concatenate()([X_a, X_b, X_c, X_d])  #

    d_neurons = 256

    X = tf.keras.layers.Dense(d_neurons, "relu")(concat)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    # X = tf.keras.layers.Dense(d_neurons, "relu")(X)

    # concat = tf.keras.layers.Lambda(lambda x: x / 500)(concat)  # get those 350_000 neurons to count for about 700

    # X = tf.keras.layers.Concatenate()([concat, X])

    X = tf.keras.layers.Dense(depth)(X)

    model = tf.keras.models.Model(inputs=[X_conv_a, X_conv_b, X_conv_c, X_conv_d], outputs=X, name='SortOfResNet50')
    return model