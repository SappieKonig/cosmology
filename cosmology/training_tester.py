import numpy as np
from model import make_model
import tqdm
import tensorflow as tf
from personal_library.personal_lib import animator
import time
import threading
import matplotlib.pyplot as plt

tf.keras.mixed_precision.set_global_policy("float32")
dtype = tf.float32

# the following loads a model from memory to train on
# model = tf.keras.models.load_model("/home/ignace/tensorTestModels/please_just_work_for_once_2D_v2")

# we have 4 different radii for different levels of precision
r_a = 40
r_b = 20
r_c = 10
r_d = 5

# the np.load() function loads the dataset into memory for further use
mass = np.load("C:/datasets/16384x16384x10_second_lmstellar.npy")
slice = mass[16384//3: 2*16384//2, :16384//2]
mass = (mass - np.average(slice)) / np.std(slice)
conv = np.load("C:/datasets/16384x16384x10_second_kappa.npy")
guess = 0
# indices = conv != 0
# conv[indices] = (conv[indices] - np.average(conv[indices])) / np.std(indices)
# print(np.count_nonzero(conv))
indices = np.load("C:/datasets/indices.npy")
size = 16384

# since we need to be able to pick a pixel that has enough room around it for context,
# we can not pick from the edges of our map, since there will not be enough room left
# this code trims the possible galaxies we can choose from to only include those with extra headroom for context
ra_min = int(31 / 90 * size)
ra_max = int(59 / 90 * size)

dec_min = int(1 / 90 * size)
dec_max = int(29 / 90 * size)

"""
een test dat alles klopt

field = np.zeros((size, size))
field[indices[:, 0], indices[:, 1]] = 1
step = 1024
img = np.zeros((16, 16))
for i in range(0, size, step):
    for j in range(0, size, step):
        sum = np.sum(field[i: i+1024, j: j+1024])
        img[int(i/step), int(j/step)] = sum

plt.imshow(img)
plt.show()

time.sleep(100000)
"""

valid_indices = (indices[:, 0] >= r_a + ra_min) * (indices[:, 0] < ra_max - r_a) * (indices[:, 1] >= r_a + dec_min) * (
        indices[:, 1] < dec_max - r_a)

indices = indices[valid_indices]
x = np.arange(len(indices))
np.random.shuffle(x)
split = int(.8 * len(x))
train, validate = x[:split], x[split:]

print(len(indices))

val_indices = indices[validate]
indices = indices[train]

print(len(val_indices), len(indices))

depth = 10
dia_a = 2 * r_a + 1
dia_b = 2 * r_b + 1
dia_c = 2 * r_c + 1
dia_d = 2 * r_d + 1

# these long lists of code return the indices of the context for the four different radii
# so let's say the context diameter is 3, and our pixel is at position (1, 1)
# then as context, we also want the pixels at positions:
# (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
# this piece of code selects those pixels

ran = np.tile(np.arange(dia_a), dia_a)
var1_a = ran.reshape((dia_a, dia_a), order="F").flatten()
var2_a = ran.reshape((dia_a, dia_a), order="C").flatten()
options1_a = np.array([var1_a + i + r_a - r_a for i in range(len(mass) - 2 * r_a)])
options2_a = np.array([var2_a + i + r_a - r_a for i in range(len(mass) - 2 * r_a)])
print(options1_a.shape)

ran = np.tile(np.arange(dia_b), dia_b)
var1_b = ran.reshape((dia_b, dia_b), order="F").flatten()
var2_b = ran.reshape((dia_b, dia_b), order="C").flatten()
options1_b = np.array([var1_b + i + r_a - r_b for i in range(len(mass) - 2 * r_a)])
options2_b = np.array([var2_b + i + r_a - r_b for i in range(len(mass) - 2 * r_a)])

ran = np.tile(np.arange(dia_c), dia_c)
var1_c = ran.reshape((dia_c, dia_c), order="F").flatten()
var2_c = ran.reshape((dia_c, dia_c), order="C").flatten()
options1_c = np.array([var1_c + i + r_a - r_c for i in range(len(mass) - 2 * r_a)])
options2_c = np.array([var2_c + i + r_a - r_c for i in range(len(mass) - 2 * r_a)])

ran = np.tile(np.arange(dia_d), dia_d)
var1_d = ran.reshape((dia_d, dia_d), order="F").flatten()
var2_d = ran.reshape((dia_d, dia_d), order="C").flatten()
options1_d = np.array([var1_d + i + r_a - r_d for i in range(len(mass) - 2 * r_a)])
options2_d = np.array([var2_d + i + r_a - r_d for i in range(len(mass) - 2 * r_a)])

# this make_model() function can be used when we do not have a trained model in memory
# or when we want to try out a new architecture. The function returns a model
model = make_model(r_a, r_b, r_c, r_d, depth)
# model = make_model(r_c, depth)
# model = tf.keras.models.load_model("C:/tensorTestModels/please_just_work_for_once_2D_v3")

# the optimizer decides how the weights are udpated
# the parameter given to the optimizer is the learning rate, in this case 3e-4
optimizer = tf.keras.optimizers.Adam(3e-4)
# optimizer = model.optimizer
# model = make_model(r_a, r_b, r_c, r_d, depth)

# configuring the model
model.compile(optimizer)
model.summary()

save_path = "C:/tensorTestModels/please_just_work_for_once_2D_v5"

# with neural networks, one epoch means going through the entire dataset once
# since we do not have that amount of time, we define an epoch as being 100.000
# entries from the dataset
epoch_size = len(indices)

# the animator shows the progress of the neural network, by graphing the value of the MSE
an = animator.Animator()


def get_piece(l, index, choices):
    l[index] = tf.cast(mass[options1_b[choices[:, 0] - r_a], options2_b[choices[:, 1] - r_a]].reshape(-1,
                                                                                                       dia_b,
                                                                                                       dia_b,
                                                                                                       depth), dtype)

for _ in range(10000):

    # selects 100.000 random entries from the dataset
    np.random.shuffle(indices)

    choices = indices

    # to stabilize training we update the weights by calculating the average desired change over 64 entries
    batch_size = 256

    # the amount of batches is determined by how many batches can fit into the size of one epoch. The // operator divides and rounds down.
    batches = np.math.ceil(epoch_size / batch_size)
    print(batches)

    # an array to keep track of the loss over time
    loss_arr = np.zeros(batches)
    naive_arr = np.zeros(batches)
    loss_count = np.zeros(batches)
    t = 0

    collection = None
    batch_choices = None

    p_bar = tqdm.trange(batches, leave=True)
    for batch in p_bar:

        if batch % 1000 == 999:
            tf.keras.models.save_model(model, save_path)

        # select the dataset entries belonging to this batch

        # these four variables grab the context from the grid with decreasing radii

        batch_choices = choices[batch * batch_size:(batch + 1) * batch_size, :]

        if batch % 30 == 0 and 0:
            collection = [0 for _ in range(30)]
            batch_choices = [choices[(batch + x) * batch_size:(batch + x + 1) * batch_size, :] for x in range(30)]
            threads = []
            for i in range(30):
                thread = threading.Thread(get_piece(collection, i, batch_choices[i]))
                thread.start()
                threads += [thread]

            for thread in threads:
                thread.join()

        net_input_a = tf.cast(
            mass[options1_a[batch_choices[:, 0] - r_a], options2_a[batch_choices[:, 1] - r_a]].reshape(-1,
                                                                                                       dia_a,
                                                                                                       dia_a,
                                                                                                       depth), dtype)
        # net_input_b = tf.cast(
        #     mass[options1_b[batch_choices[:, 0] - r_a], options2_b[batch_choices[:, 1] - r_a]].reshape(-1,
        #                                                                                                dia_b,
        #                                                                                                dia_b,
        #                                                                                                depth), dtype)
        # net_input_b = collection[batch % 30]
        # net_input_b = tf.cast(mass[options1_b[batch_choices[:, 0] - r_a], options2_b[batch_choices[:, 1] - r_a]].reshape(-1,
        #                                                                                        dia_b,
        #                                                                                        dia_b,
        #                                                                                        depth), dtype)
        # net_input_c = tf.cast(
        #     mass[options1_c[batch_choices[:, 0] - r_a], options2_c[batch_choices[:, 1] - r_a]].reshape(-1,
        #                                                                                                dia_c,
        #                                                                                                dia_c,
        #                                                                                                depth), dtype)
        # net_input_d = tf.cast(
        #     mass[options1_d[batch_choices[:, 0] - r_a], options2_d[batch_choices[:, 1] - r_a]].reshape(-1,
        #                                                                                                dia_d,
        #                                                                                                dia_d,
        #                                                                                                depth), dtype)



        # net_output = tf.cast(conv[batch_choices[batch % 30][:, 0], batch_choices[batch % 30][:, 1]].reshape(-1, depth), dtype)
        net_output = tf.cast(conv[batch_choices[:, 0], batch_choices[:, 1]].reshape(-1, depth), dtype)


        # the desired output, the true convergence, is taken from the convergence grid

        start = time.time()
        with tf.GradientTape() as tape:
            # the model predicts the convergence by processing the input
            # output = model([net_input_a, net_input_b, net_input_c, net_input_d])
            output = model(net_input_a)
            # output = np.zeros(net_output.shape)
            # the mean square error is computed
            loss = (net_output - output) ** 2

            # because most of the pixels do not contain a value for convergence,
            # as there are 5 times as many pixels as galaxies in the dataset,
            # we have to remove the pixels without convergence from the loss, as to not bias our network
            loss = tf.where(net_output != 0, loss, 0.)


        # lets tensorflow calculate and process the gradients with respect to the loss
        train_vars = model.trainable_variables
        grads = tape.gradient(loss, train_vars)

        optimizer.apply_gradients(zip(grads, train_vars))
        t += time.time() - start

        # saves the loss, so it can later be displayed to show how good the network is doing
        loss_arr[batch] = np.sum(loss)
        naive_arr[batch] = np.sum((net_output - guess) ** 2)
        loss_count[batch] = np.count_nonzero(loss)

        loss = np.sum(loss_arr) / np.sum(loss_count)
        naive_loss = np.sum(naive_arr) / np.sum(loss_count)
        p_bar.set_description(f"loss_fraction: {loss/naive_loss}", refresh=True)
        # loss = np.sum(naive_arr) / np.sum(loss_count)
        # print("naive:", loss)

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
        # select the dataset entries belonging to this batch
        batch_choices = choices[batch * batch_size:(batch + 1) * batch_size, :]

        # these four variables grab the context from the grid with decreasing radii
        # net_input_a = tf.cast(
        #     mass[options1_a[batch_choices[:, 0] - r_a], options2_a[batch_choices[:, 1] - r_a]].reshape(-1,
        #                                                                                                dia_a,
        #                                                                                                dia_a,
        #                                                                                                depth), dtype)
        net_input_b = tf.cast(
            mass[options1_b[batch_choices[:, 0] - r_a], options2_b[batch_choices[:, 1] - r_a]].reshape(-1,
                                                                                                       dia_b,
                                                                                                       dia_b,
                                                                                                       depth), dtype)
        # net_input_c = tf.cast(
        #     mass[options1_c[batch_choices[:, 0] - r_a], options2_c[batch_choices[:, 1] - r_a]].reshape(-1,
        #                                                                                                dia_c,
        #                                                                                                dia_c,
        #                                                                                                depth), dtype)
        # net_input_d = tf.cast(
        #     mass[options1_d[batch_choices[:, 0] - r_a], options2_d[batch_choices[:, 1] - r_a]].reshape(-1,
        #                                                                                                dia_d,
        #                                                                                                dia_d,
        #                                                                                                depth), dtype)

        # the desired output, the true convergence, is taken from the convergence grid
        net_output = tf.cast(conv[batch_choices[:, 0], batch_choices[:, 1]].reshape(-1, depth), dtype)
        start = time.time()
        # the model predicts the convergence by processing the input
        # output = model([net_input_a, net_input_b, net_input_c, net_input_d])
        output = model(net_input_b)
        # output = np.zeros(net_output.shape)
        # the mean square error is computed
        loss = (net_output - output) ** 2

        # because most of the pixels do not contain a value for convergence,
        # as there are 5 times as many pixels as galaxies in the dataset,
        # we have to remove the pixels without convergence from the loss, as to not bias our network
        loss = tf.where(net_output != 0, loss, 0.)

        t += time.time() - start

        # saves the loss, so it can later be displayed to show how good the network is doing
        loss_arr[batch] = np.sum(loss)
        naive_arr[batch] = np.sum((net_output - guess) ** 2)
        loss_count[batch] = np.count_nonzero(loss)

        loss = np.sum(loss_arr) / np.sum(loss_count)
        naive_loss = np.sum(naive_arr) / np.sum(loss_count)
        p_bar.set_description(f"val_loss_fraction: {loss / naive_loss}", refresh=True)

    # show the average loss over the entire epoch
    loss = np.sum(loss_arr) / np.sum(loss_count)
    print("val_loss:", loss)
    loss = np.sum(naive_arr) / np.sum(loss_count)
    print("val_naive:", loss)
    print(t)

    # makes a graph of the loss, to show whether the model is still improving
    # an.push(loss)
    tf.keras.models.save_model(model, save_path)
