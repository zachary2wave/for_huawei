
import keras as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Model, model_from_config
from matplotlib import pyplot as plt



class Energy():
    def __init__(self, I, Io, Nchannel, NAP, Nmodel, **kwargs):
        self.I = I
        self.Io = Io
        self.Nchannel = Nchannel
        self.Nmodel = Nmodel
        self.NAP = NAP
        self.Wight = []
        self.G = []
        for i in range(NAP):
            for j in range(i + 1, NAP):
                self.Wight.append(self.I[i, j]+self.I[j, i])
        self.I = self.I[np.newaxis, :, :, np.newaxis]
        self.train_fn_list = []



    def clone_model(self, model, custom_objects={}):
        # Requires Keras 1.0.7 since get_config has breaking changes.
        config = {
            'class_name': model.__class__.__name__,
            'config': model.get_config(),
        }
        clone = model_from_config(config, custom_objects=custom_objects)
        clone.set_weights(model.get_weights())
        return clone

    def setup_model(self, conv_layers, conv_kernel_size, dense_layer, dropout):
        input = Input(shape=(self.NAP, self.NAP, 1))
        for layers, kernel_size in zip(conv_layers, conv_kernel_size):
            X = Conv2D(layers, kernel_size, padding='same', activation="relu")(
                input)
        X = Flatten()(X)
        for layer in dense_layer:
            X = Dense(layer, activation="relu")(X)
            if dropout:
                X = Dropout(0.5)(X)
        Xzi = []
        for i in range(self.NAP):
            Xzi.append(Dense(self.Nchannel, activation="softmax")(X))
        model = Model(inputs=input, outputs=Xzi)
        return model

    def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
        with tf.variable_scope(scope):
            nin = x.get_shape()[1].value
            w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
            b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
            return tf.matmul(x, w) + b


    def setup_model(self, conv_layers, conv_kernel_size, dense_layer, dropout):
        cluster = tf.placeholder(shape=(self.NAP * self.Nfeature, 1))
        input = Input(shape=(self.NAP*self.Nfeature, 1))
        input = tf.matmul(input,cluster)
        X = Dense(128, activation="relu")(input)

        for flag, den in enumerate(dense_layer):
            with tf.name_scope("la"+str(flag)):
                kernel = tf.reshape(tf.constant([2., 2.]), shape=[1, 2])
                bias = tf.reshape(tf.constant([1., 1., 1., 1., 1., 1., 1., 1.]), shape=[4, 2])
                b2 = tf.add(tf.matmul(input, kernel), bias)

        for layers, kernel_size in zip(conv_layers, conv_kernel_size):
            X = Conv2D(layers, kernel_size, padding='same', activation="relu")(
                input)
        X = Flatten()(X)
        for layer in dense_layer:

            if dropout:
                X = Dropout(0.5)(X)
        Xzi = []
        for i in range(self.NAP):
            Xzi.append(Dense(self.Nchannel, activation="softmax")(X))
        model = Model(inputs=input, outputs=Xzi)










    def energy(self, label):
        energylist = []
        flag = 0
        for AP1 in range(self.NAP):
            for AP2 in range(AP1 + 1, self.NAP):
                templabel1 = label[AP1]
                templabel2 = label[AP2]
                energylist.append(tf.reduce_sum(tf.multiply(templabel1, templabel2))*self.Wight[flag])
                flag += 1
        energy_inside = tf.reduce_sum(energylist)
        energylist = []
        for AP1 in range(self.NAP):
            energylist.append(tf.reduce_sum(self.Io[AP1, :] * label[AP1]))
        energy_outside = tf.reduce_sum(energylist)

        return energy_inside + 0 * energy_outside



    def setup_fn(self, model, noise_f):
        noise = tf.compat.v1.placeholder(dtype=tf.float32)
        "备用"
        # lrinput = tf.placeholder(dtype=tf.float32)
        # lr = tf.get_variable(dtype=tf.float32, shape=[1], initializer=1e-3)
        # tf.assign(lr, lrinput)
        opt = K.optimizers.Adagrad(learning_rate=1e-4)
        updates = opt.get_updates(params=model.trainable_weights,
                                  loss=self.energy(model.output)+noise_f*tf.random.normal(shape=[1], mean=0., stddev=noise))
        train_fn = K.backend.function([model.input, noise], [model.output, self.energy(model.output)],
                                       updates=updates)
        return train_fn

    def compile(self, model_parameter_list):
        N_model = len(model_parameter_list)
        assert N_model == self.Nmodel
        model = []
        for model_loop in range(self.Nmodel):
            print(model_loop)
            # self.G.append(tf.Graph())
            # with self.G[-1].as_default():
            with tf.name_scope(name="model"+str(model_loop)):
                model.append(self.setup_model(model_parameter_list[model_loop]["conv_layers"],
                                              model_parameter_list[model_loop]["conv_kernel_size"],
                                              model_parameter_list[model_loop]["dense_layer"],
                                              model_parameter_list[model_loop]["dropout"]))
                self.train_fn_list.append(self.setup_fn(model[-1], model_parameter_list[model_loop]["noise"]))
        return model


    def train(self, times,
              noise_start=10, noise_decay=0.1, noise_decay_time=1000,
              lr_start=1e-3, lr_decay=0.1, lr_decay_time=3000):
        num_model = len(self.train_fn_list)
        assert num_model == self.Nmodel
        loss = np.zeros([times, num_model])
        output = []
        record_mean_loss = []
        for loop_time in range(times):
            noise = noise_start * noise_decay ** (loop_time / noise_decay_time)
            inputs = self.I + 0.01 * noise * np.random.randn(1, self.NAP, self.NAP, 1)
            # lr_input = lr_start * lr_decay ** (loop_time / lr_decay_time)   # 备用
            for loop_model in range(num_model):
                # graph = self.G[loop_model]
                # with graph.as_default():
                #     K.backend.set_session(sess)
                feedin = {"inputs": inputs, "noise": noise}
                o, lt = self.train_fn_list[loop_model](feedin)
                loss[loop_time, loop_model] = lt
            record_mean_loss.append(sum(loss[loop_time, :]))
            print("times = ", loop_time, "\t", "loss_mean", record_mean_loss[-1])
        for loop_model in range(num_model):
            feedin = {"inputs": inputs, "noise": 0}
            o, lt = self.train_fn_list[loop_model](feedin)
            output.append(o)
        return loss, output, record_mean_loss

    def analysis(self, output):
        APchannel = []
        for i in range(self.NAP):
            APchannel.append(np.argmax(output[i]))
        channelset = []
        for j in range(self.Nchannel):
            channelset.append([i for i in range(len(APchannel)) if APchannel[i] == j])
        return channelset
