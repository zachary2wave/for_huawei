import keras as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten
from keras.models import Model
from matplotlib import pyplot as plt
import time
NAP = 15            # AP 数目
Nchannel = 15       # 信道 数目

# sess = tf.Session()
# logdir = time.strftime("%Y%m%d_%H_%M", time.localtime())+"/"
# writer = tf.summary.FileWriter(logdir, sess.graph)
# Lossrecord = tf.placeholder(tf.float32, [])
# tf.summary.scalar("Loss", Lossrecord)


total_train_time = 1000



# the interference matrix
" better to normalize "
Interference_matrix = np.zeros([NAP, NAP])
for i in range(NAP):
    for j in range(i+1, NAP):
        randomnum = np.random.uniform(0, 1)
        Interference_matrix[i, j] = randomnum
        Interference_matrix[j, i] = randomnum
# print(Interference_matrix)
class Energy():
    def __init__(self, I, Nchannel, NAP, model):
        self.I = I
        self.Nchannel = Nchannel
        self.NAP = NAP
        self.Wight = []
        for i in range(NAP):
            for j in range(i + 1, NAP):
                self.Wight.append(self.I[i, j])
        self.model = model
        self.I = self.I[np.newaxis, :, :, np.newaxis]

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

    def compile(self, lr):
        opt = K.optimizers.Adagrad(learning_rate=lr)
        updates = opt.get_updates(params=self.model.trainable_weights, loss=self.energy(self.model.output))
        self.train_fn = K.backend.function([self.model.input],  [self.model.output,self.energy(self.model.output)], updates=updates)

    def train(self, times):
        record_loss = []
        for time in range(times):
            inputs = self.I + 0.001 * np.random.randn(1, NAP, NAP,1)
            outputs, loss = self.train_fn(inputs)
            record_loss.append(loss)
            APchannel = []
            for i in range(self.NAP):
                APchannel.append(np.argmax(outputs[i]))
            channelset=[]
            for j in range(Nchannel):
                channelset.append([i for i in range(len(APchannel)) if APchannel[i] == j])
            print("times = ", time, "\t", "loss", loss)
        print(channelset)
        return record_loss



# 备用
# def clipped_error(X, label):
#     error = energy(X, label)
#     if error > 10:
#         return 10
#     else:
#         return error

# setup the network
input= Input(shape=(NAP, NAP, 1))
X = Conv2D(32, (3, 3), padding='same', activation="relu")(input)
X = Conv2D(32, (3, 3), padding='same', activation="relu")(X)
X = Flatten()(X)
X = Dense(Nchannel*NAP, activation="relu")(X)
Xzi = []
for i in range(NAP):
    Xzi.append(Dense(Nchannel, activation="softmax")(X))
# out = concatenate(Xzi)
model = Model(inputs=input, outputs=Xzi)

agent = Energy(Interference_matrix, NAP=NAP, Nchannel=Nchannel, model=model)
agent.compile(lr=1e-4)
record_loss = agent.train(10000)
plt.figure()
plt.plot(record_loss)
plt.title("training_loss")
plt.show()
