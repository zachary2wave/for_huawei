import keras as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten, MaxPooling2D
from keras.models import Model, model_from_config
from matplotlib import pyplot as plt
import time
NAP = 15           # AP num
Nchannel = 4       # channel num
total_train_time = 1000


" 最好进行归一化。"
Interference_matrix = np.zeros([NAP, NAP])
for i in range(NAP):
    for j in range(i+1, NAP):
        randomnum = np.random.uniform(0, 1)
        Interference_matrix[i, j] = randomnum
        Interference_matrix[j, i] = randomnum


"并行model"
def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


def setupmodel(conv_layers, conv_kernel_size, pooling, dense_layer):
    for layers in conv_layers:
        X = Conv2D(layers, conv_kernel_size, padding='same', activation="relu")(
            input)
        if pooling:
            X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(X)
    X = Flatten()(X)
    for layer in dense_layer:
        X = Dense(layer, activation="relu")(X)
    Xzi = []
    for i in range(NAP):
        Xzi.append(Dense(Nchannel, activation="softmax")(X))
    model = Model(inputs=input, outputs=Xzi)
    return model


class Energy():
    def __init__(self, I, Nchannel, NAP):
        self.I = I
        self.Nchannel = Nchannel
        self.NAP = NAP
        self.Wight = []
        for i in range(NAP):
            for j in range(i + 1, NAP):
                self.Wight.append(self.I[i, j])
        self.I = self.I[np.newaxis, :, :, np.newaxis]
        self.train_fn_list=[]

    def energy(self,label):
        energylist = []
        flag = 0
        for AP1 in range(NAP):
            for AP2 in range(AP1 + 1, NAP):
                templabel1 = label[AP1]
                templabel2 = label[AP2]
                energylist.append(tf.reduce_sum(tf.multiply(templabel1, templabel2))*self.Wight[flag])
                flag += 1
        energy = tf.reduce_sum(energylist)
        return energy

    def setupfn(self, model):
        noise = tf.placeholder(dtype=tf.float32)
        lr = tf.placeholder(dtype=tf.float32)
        opt = K.optimizers.Adagrad(learning_rate=lr)
        updates = opt.get_updates(params=model.trainable_weights,
                                  loss=self.energy(model.output)+tf.random_normal(shape=[1], mean=0., stddev=noise))
        train_fn = K.backend.function([model.input, noise, lr], [model.output, self.energy(model.output)],
                                       updates=updates)
        return train_fn

    def com_energy(self,label):
        num_model=len(label)
        for i in range(num_model):



    def compile(self, model_parameter_list):
        num_model = model_parameter_list.shape
        input = Input(shape=(NAP, NAP, 1))
        output = []
        for loop_model in range(num_model):
            model_parameter_list["conv_layers"] = [[32, 32], [32, 32], [32, 32, 32], [32, 32]]
            model_parameter_list["conv_kernel_size"] = [[5, 5], [5, 5], [7, 7], [3, 3]]
            model_parameter_list["pooling"] = [0, 0, 0, 0]
            model_parameter_list["pooling_kernel"] = [0, 0, 0, 0]
            model_parameter_list["dense_layer"] = [[128, 128], [128, 128], [128], [128]]
            with tf.name_scope("model"+str(time)):
                for layers in model_parameter_list["conv_layers"][loop_model]:
                    X = Conv2D(layers, model_parameter_list["conv_kernel_size"][loop_model], padding='same', activation="relu")(input)
                    if model_parameter_list["pooling"][loop_model]:
                        X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(X)
                X = Flatten()(X)
                for layer in model_parameter_list["dense_layer"][loop_model]:
                    X = Dense(layer, activation="relu")(X)
                Xzi = []
                for i in range(NAP):
                    Xzi.append(Dense(Nchannel, activation="softmax")(X))
            output.append(Xzi)
        model = Model(inputs=input, outputs=output)
        model.summary()

        noise = tf.placeholder(dtype=tf.float32)
        lr = tf.placeholder(dtype=tf.float32)
        opt = K.optimizers.Adagrad(learning_rate=lr)
        updates = opt.get_updates(params=model.trainable_weights,
                                  loss=self.energy(model.output)+tf.random_normal(shape=[1], mean=0., stddev=noise))
        self.train_fn = K.backend.function([model.input, noise, lr], [model.output, self.energy(model.output)],
                                       updates=updates)

    def train(self, times,
              noise_start=0.1, noise_decay=0.1, noise_decay_time=1000,
              lr_start=1e-3, lr_decay=0.1, lr_decay_time=3000):
        num_model = len(self.train_fn_list)
        loss = np.zeros([times, num_model])
        output = np.zeros([num_model, self.NAP*self.Nchannel])
        record_mean_loss = []
        for loop_time in range(times):
            inputs = self.I + 0.001 * np.random.randn(1, NAP, NAP,1)
            noise = noise_start * noise_decay ** (loop_time / noise_decay_time)
            lr = lr_start * lr_decay ** (loop_time / lr_decay_time)
            for loop_model in range(num_model):
                o, lt = self.train_fn_list[loop_model](inputs, noise, lr)
                loss[loop_time, loop_model] = lt
            record_mean_loss.append(sum(loss[loop_time, :]))
            print("times = ", time, "\t", "loss_mean", record_mean_loss[-1])
        for loop_model in range(num_model):
            o, lt = self.train_fn_list[loop_model](inputs, noise, lr)
            output[loop_model, :] = o
        return loss, output, record_mean_loss
# 备用
# def clipped_error(X, label):
#     error = energy(X, label)
#     if error > 10:
#         return 10
#     else:
#         return error


agent = Energy(Interference_matrix, NAP=NAP, Nchannel=Nchannel)

"setup network"
model_parameter_list = dict()
model_parameter_list["conv_layers"] = [[32, 32], [32, 32], [32, 32, 32], [32, 32]]
model_parameter_list["conv_kernel_size"] = [(5, 5), (3, 3), (3, 3), (7, 7)]
model_parameter_list["pooling"] = [0, 0, 0, 0]
model_parameter_list["pooling_kernel"] = [(2, 2), (3, 3), (2, 2), (2, 2)]
model_parameter_list["dense_layer"] = [[128, 128], [128, 128], [128], [128]]

agent.compile(model_parameter_list)
loss, output, record_mean_loss = agent.train(10000)
" showing the loss"
plt.figure()
plt.plot(record_mean_loss)
plt.title("training_loss")
plt.show()

"combine the output "

output



