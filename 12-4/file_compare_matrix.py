#%%
import pandas
import numpy as np

I = pandas.read_csv("1F_matrix.csv")
I = np.array(I.iloc[:,1:])
# I = (I + I.T)/2
IP = pandas.read_csv("cu_1F.csv")
channel_list = list(set(IP.channel))
IP = np.array(IP.iloc[:,3])
outsideIF = pandas.read_csv("outside_infer.csv")
outsideIF = np.array(outsideIF.iloc[:,1:])

APname = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20,
 21, 23, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
 47, 48, 49, 50, 60, 61, 62, 63, 64, 65, 66, 67]

#%%
"参数部分"
NAP = I.shape[0]
Nchannel = len(channel_list)
Nmodel=1

#%%
def analysis_output(channelset):
    set_interference = np.zeros_like(channelset)
    for channel_loop, channeltemp in zip(range(len(channelset)),channelset):
        Ntemp = len(channeltemp)
        for i in range(Ntemp):
            for j in range(i+1, Ntemp):
                set_interference[channel_loop] += I[channeltemp[i], channeltemp[j]]
                set_interference[channel_loop] += I[channeltemp[j], channeltemp[i]]
    for time in range(len(channelset)):
        print(channelset[time], set_interference[time],"\n")
    channel_set=[]
    for tpset in channelset:
        tpsss = []
        for tpap in tpset:
            tpsss.append(APname[tpap])
        channel_set.append(tpsss)
    return channel_set, set_interference, sum(set_interference)

#%%
'''
 DEN 网络部分
'''
#%%
from DEN import Energy

agent = Energy(I, outsideIF, NAP=NAP, Nchannel=Nchannel, Nmodel=Nmodel)
model_parameter_list = []
for i in range(Nmodel):
    model_parameter = dict()
    model_parameter["conv_layers"] = [32, 64, 128]
    model_parameter["conv_kernel_size"] = [(7, 7), (5, 5), (3, 3)]
    model_parameter["dense_layer"] = [128, 128, 128]
    model_parameter["noise"] = [1]
    model_parameter_list.append(model_parameter)
agent.compile(model_parameter_list)

#%%
loss, output, record_mean_loss = agent.train(1000)
#%%
channel_set = agent.analysis(output[0])
channelset_den, set_interference, total_I = analysis_output(channel_set)

#%%

import matplotlib.pyplot as plt
plt.figure()
for time in range(Nmodel):
    plt.plot(loss[:, time], label="model"+str(time))
    plt.legend()
plt.show()
#%%
'''
 谱聚类方式
'''
from SPC import Spectrum

sp = Spectrum(n_cluster=Nchannel, method='normalized_Lsym', criterion='gaussian', gamma=2)
sp.fit((I.T + I)/2)
#%%
cluster = sp.cluster
listchannel = list(cluster.values())
channelset_sp, set_interference, total_I = analysis_output(listchannel)

#%%
from figure_show import show
show(channelset_den)
plt.show()
show(channelset_sp)
plt.show()