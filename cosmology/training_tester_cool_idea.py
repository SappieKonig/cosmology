import numpy as np
from model import make_model
import tqdm
import tensorflow as tf
from personal_library.personal_lib import animator
import time
import sys

# np.set_printoptions(threshold=sys.maxsize)

tf.keras.mixed_precision.set_global_policy("float32")
dtype = tf.float32

r = 5

indices_grid = np.load("C:/datasets/16384x16384x10_index_lmstellar.npy")

# indices = np.load("C:/datasets/indices.npy")
pos_kappa_mass_by_index = np.load("C:/datasets/pos_kappa_mass_by_index.npy")

size = 16384

depth = 10
dia = 2 * r + 1
input_len = 128

ra_index = (pos_kappa_mass_by_index[:, 0] / 90 * size).astype(np.int32)
dec_index = (pos_kappa_mass_by_index[:, 1] / 90 * size).astype(np.int32)
d_z = 1.41708 - 0.07296

# get the indexes by redistributing it from 0.07296-1.41708 to 0-9
z_index = ((pos_kappa_mass_by_index[:, 2] - .07296) / d_z * 10).astype(np.int32)
indices = np.stack([ra_index, dec_index, z_index], axis=1)

ra_min = int(31 / 90 * size)
ra_max = int(59 / 90 * size)

dec_min = int(1 / 90 * size)
dec_max = int(29 / 90 * size)

valid_indices = (indices[:, 0] >= r + ra_min) * (indices[:, 0] < ra_max - r) * (indices[:, 1] >= r + dec_min) * (
        indices[:, 1] < dec_max - r) * (indices[:, 2] >= 0) * (indices[:, 2] < depth)

used_indices = np.arange(len(indices))[valid_indices]

x = np.arange(len(used_indices))
np.random.seed(0)
np.random.shuffle(x)
split = int(.8 * len(x))
train, validate = x[:split], x[split:]

val_indices = used_indices[validate]
train_indices = used_indices[train]

ran = np.tile(np.arange(dia), dia)
var1 = ran.reshape((dia, dia), order="F").flatten()
var2 = ran.reshape((dia, dia), order="C").flatten()
options1 = np.array([var1 + i for i in range(size - 2 * r)])
options2 = np.array([var2 + i for i in range(size - 2 * r)])

model = make_model(input_len, 1)
# model = tf.keras.models.load_model("C:/tensorTestModels/please_just_work_for_once_2D_v3")


optimizer = tf.keras.optimizers.Adam(3e-4)
# optimizer = model.optimizer


# model.compile(optimizer)
# model.summary()

save_path = "C:/tensorTestModels/please_just_work_for_once_2D_v6"
model.load_weights(save_path)


def process(choices):
    parts = indices_grid[options1[indices[choices, 0] - r], options2[indices[choices, 1] - r]].reshape(-1,
                                                                                                       dia,
                                                                                                       dia,
                                                                                                       depth)

    host_indices = indices_grid[indices[choices, 0], indices[choices, 1], indices[choices, 2]]
    records = pos_kappa_mass_by_index[host_indices]

    inputs, outputs = [], []
    for part, record in zip(parts, records):
        _in, _out = in_n_out_from_record(part, record)
        inputs += [_in]
        outputs += [_out]
    inputs = np.array(inputs)
    outputs = np.array(outputs).reshape((-1, 1))
    return inputs, outputs


def in_n_out_from_record(shard, record):
    uniques = pos_kappa_mass_by_index[shard[shard != -1]]
    distance = np.sum((uniques[:, [0, 1]] - record[[0, 1]]) ** 2, axis=-1)
    candidats = np.argsort(distance)[:input_len]
    winners = uniques[candidats]
    winners[:, [0, 1, 2]] -= record[[0, 1, 2]]
    input_part, output_part = winners[:, [0, 1, 2, 4]], record[3]
    padded = np.zeros((input_len, 5))
    padded[:len(input_part), :4] += input_part
    padded[:, -1] = record[2]
    return padded, output_part


for _ in range(10000):

    indexs = np.arange(len(train_indices))
    np.random.shuffle(indexs)

    choices = train_indices[indexs]

    batch_size = 2048

    epoch_size = len(choices)

    batches = np.math.ceil(epoch_size / batch_size)

    loss_arr = np.zeros(batches)
    naive_arr = np.zeros(batches)
    loss_count = np.zeros(batches)
    t = 0

    p_bar = tqdm.trange(batches, leave=True)
    for batch in p_bar:

        if batch % 1000 == 999:
            model.save_weights(save_path)

        # the desired output, the true convergence, is taken from the convergence grid
        net_input, net_output = process(choices[batch * batch_size:(batch + 1) * batch_size])
        start = time.time()
        with tf.GradientTape() as tape:
            # the model predicts the convergence by processing the input
            # output = model([net_input_a, net_input_b, net_input_c, net_input_d])
            output = model(net_input)
            # output = np.zeros(net_output.shape)
            # the mean square error is computed
            loss = (net_output - output) ** 2

            # because most of the pixels do not contain a value for convergence,
            # as there are 5 times as many pixels as galaxies in the dataset,
            # we have to remove the pixels without convergence from the loss, as to not bias our network

        # lets tensorflow calculate and process the gradients with respect to the loss
        train_vars = model.trainable_variables

        grads = tape.gradient(loss, train_vars)

        optimizer.apply_gradients(zip(grads, train_vars))
        t += time.time() - start

        # saves the loss, so it can later be displayed to show how good the network is doing
        loss_arr[batch] = np.sum(loss)
        naive_arr[batch] = np.sum((net_output) ** 2)
        loss_count[batch] = np.count_nonzero(loss)

        loss = np.sum(loss_arr) / np.sum(loss_count)
        naive_loss = np.sum(naive_arr) / np.sum(loss_count)
        p_bar.set_description(f"loss_fraction: {loss / naive_loss}, {loss_arr[batch] / naive_arr[batch]}, {loss}",
                              refresh=True)

    # show the average loss over the entire epoch
    loss = np.sum(loss_arr) / np.sum(loss_count)
    print("loss:", loss)
    naive_loss = np.sum(naive_arr) / np.sum(loss_count)
    print("naive:", naive_loss)
    print(t)

    choices = val_indices

    # to stabilize training we update the weights by calculating the average desired change over 64 entries

    # the amount of batches is determined by how many batches can fit into the size of one epoch. The // operator divides and rounds down.
    batches = np.math.ceil(len(val_indices) / batch_size)

    # an array to keep track of the loss over time
    loss_arr = np.zeros(batches)
    naive_arr = np.zeros(batches)
    loss_count = np.zeros(batches)
    t = 0

    p_bar = tqdm.trange(batches)
    for batch in p_bar:
        # the desired output, the true convergence, is taken from the convergence grid
        net_input, net_output = process(choices[batch * batch_size:(batch + 1) * batch_size])
        start = time.time()
        output = model(net_input)

        t += time.time() - start

        # saves the loss, so it can later be displayed to show how good the network is doing
        loss = (net_output - output) ** 2

        loss_arr[batch] = np.sum(loss)
        naive_arr[batch] = np.sum((net_output) ** 2)
        loss_count[batch] = np.count_nonzero(loss)

        loss = np.sum(loss_arr) / np.sum(loss_count)
        naive_loss = np.sum(naive_arr) / np.sum(loss_count)
        p_bar.set_description(f"loss_fraction: {loss / naive_loss}, {loss_arr[batch] / naive_arr[batch]}, {loss}",
                              refresh=True)

    # show the average loss over the entire epoch
    loss = np.sum(loss_arr) / np.sum(loss_count)
    print("val_loss:", loss)
    loss = np.sum(naive_arr) / np.sum(loss_count)
    print("val_naive:", loss)
    print(t)

    # makes a graph of the loss, to show whether the model is still improving
    # an.push(loss)
    model.save_weights(save_path)
