#%%
import keras as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten, MaxPooling2D
from keras.models import Model, model_from_config
from matplotlib import pyplot as plt
import time
NAP = 15           # AP num
Nchannel = 4       # channel num
Nmodel = 5
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
    input = Input(shape=(NAP, NAP, 1))
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

def analysis(outputs):
    APchannel = []
    for i in range(NAP):
        APchannel.append(np.argmax(outputs[i]))
    channelset=[]
    for j in range(Nchannel):
        channelset.append([i for i in range(len(APchannel)) if APchannel[i] == j])
    print(channelset)
    return channelset


class Energy():
    def __init__(self, I, Nchannel, NAP):

        global sess
        sess = tf.Session()
        self.I = I
        self.Nchannel = Nchannel
        self.NAP = NAP
        self.Wight = []
        self.G = []
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

    def setupfn(self, model, noise_label):
        noise = tf.placeholder(dtype=tf.float32)
        "备用"
        # lrinput = tf.placeholder(dtype=tf.float32)
        # lr = tf.get_variable(dtype=tf.float32, shape=[1], initializer=1e-3)
        # tf.assign(lr, lrinput)
        opt = K.optimizers.Adagrad(learning_rate=1e-4)
        updates = opt.get_updates(params=model.trainable_weights,
                                  loss=self.energy(model.output)+noise_label*tf.random_normal(shape=[1], mean=0., stddev=noise))
        train_fn = K.backend.function([model.input, noise], [model.output, self.energy(model.output)],
                                       updates=updates)
        return train_fn

    def compile(self, model_parameter_list):
        N_model = len(model_parameter_list)
        model = []
        for model_loop in range(N_model):
            print(model_loop)
            # self.G.append(tf.Graph())
            # with self.G[-1].as_default():
            with tf.name_scope(name="model"+str(model_loop)):
                model.append(setupmodel(model_parameter_list[model_loop]["conv_layers"],
                                        model_parameter_list[model_loop]["conv_kernel_size"],
                                        model_parameter_list[model_loop]["pooling"],
                                        model_parameter_list[model_loop]["dense_layer"]))
                self.train_fn_list.append(self.setupfn(model[-1], model_parameter_list[model_loop]["noise"]))

    def train(self, times,
              noise_start=0.1, noise_decay=0.1, noise_decay_time=1000,
              lr_start=1e-3, lr_decay=0.1, lr_decay_time=3000):
        num_model = len(self.train_fn_list)
        loss = np.zeros([times, num_model])
        output = []
        record_mean_loss = []
        for loop_time in range(times):
            inputs = self.I + 0.001 * np.random.randn(1, NAP, NAP,1)
            noise = noise_start * noise_decay ** (loop_time / noise_decay_time)
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
# 备用
# def clipped_error(X, label):
#     error = energy(X, label)
#     if error > 10:
#         return 10
#     else:
#         return error

#%%
agent = Energy(Interference_matrix, NAP=NAP, Nchannel=Nchannel)

"setup network"
model_parameter_list = []
for i in range(Nmodel):
    model_parameter = dict()
    model_parameter["conv_layers"] = [32, 32, 32]
    model_parameter["conv_kernel_size"] = (5, 5)
    model_parameter["dense_layer"] = [128, 128, 128]
    model_parameter["noise"] = [1]
    model_parameter_list.append(model_parameter)
agent.compile(model_parameter_list)

#%%
loss, output, record_mean_loss = agent.train(1000)

#%%

channel = []
best_result = []
for time in range(Nmodel):
    best_result.append(loss[-1,time])
    channel.append(analysis(output[time]))
print(best_result)

plt.figure()
for time in range(Nmodel):
    plt.plot(loss[:, time], label="model"+str(time))
    plt.legend()

" showing the loss"
plt.figure()
plt.plot(record_mean_loss)
plt.title("training_loss")
plt.show()


#%%
def cal_score(actual_set):
    I = 0
    for channel_loop in range(Nchannel-1):
        tp_set = list(actual_set[channel_loop])
        for i in range(len(tp_set)):
            for j in range(i+1,len(tp_set)):
                I += Interference_matrix[i, j]
    totalset = set()
    for set_loop in range(Nchannel, len(actual_set)):
         totalset = totalset | actual_set[set_loop]
    tp_set = list(totalset)
    for i in range(len(tp_set)):
        for j in range(i + 1, len(tp_set)):
            I += Interference_matrix[i, j]
    return I
#%%
"利用层次聚类投票"
Ntimes = NAP-Nchannel
now_set = [set([i]) for i in range(NAP)]
model_put = np.squeeze(np.array(output[0]))
score_model = loss[-1, :]
for time in range(1):
    Nset = len(now_set)
    ori_I = cal_score(now_set)
    " 创建空评分矩阵 "
    score = dict()
    score["index"] = []
    score["item"] = []
    score["score"] = []
    "2nd order"
    for i in range(Nset):
        for j in range(i + 1, Nset):
            score["index"].append((i, j))
            score["item"].append(now_set[i] | now_set[j])
            score["score"].append(0)
            for model_loop in range(Nmodel):
                for channel_loop in range(Nchannel):
                    tempset = channel[model_loop][channel_loop]
                    if score["item"][-1] < set(tempset):
                        score["score"][-1] += 1 / score_model[model_loop]
                        break
    "开始创建"
    score_list = score["score"]
    index = [i for i in range(len(score_list)) if score_list[i] == max(score_list)]
    first, second = score["index"][np.random.choice(index)]
    del now_set[first], now_set[second-1]
    now_set.insert(0, set(score["item"][index]))
    after_I = cal_score(now_set)
    print(now_set, "the ori I", ori_I, "the after I", after_I)





