
import numpy as np
import matplotlib.pyplot as plt

'''
该程序参考

2000 年 JSCA 论文
Performance analysis of IEEE 802.11 distributed coordination function
总体环境
# avr_ap = 1 为均匀分布  其他为 随机分布
所有的 AP 和UE都利用同样的信道 
这里需要改变载波感知范围、 功率

重点 这里只关心 下行信道 ，也就是UE不去参与竞争

'''


class scenario:

    def __init__(self, NAP, NUE, freq, avr_ap):
        'the envitonment'
        self.MAZE_H = 150
        self.MAZE_W = 150
        self.NAP = NAP
        self.NUE = NUE
        # 2.4G 参数
        self.nsf2 = 3.3    # 2.4G 同层的穿透
        self.P02 = 38
        self.alpha2 = 0.01  # 2.4G 信道衰减指数
        self.normalstd2 = 3.75  # 2.4G  标准差

        #5G 参数
        self.nsf5 = 1.582     # 5G   同城的穿透
        self.alpha2 = 0.01     # 2.4G 信道衰减指数
        self.alpha5 = 0.4     # 5G   信道衰减指数
        self.FAF2 = 13        # 2.4G 穿透
        self.FAF5 = 24        # 5G   穿透
        self.freq = freq      # 频率选择
        self.normalstd2 = 3   # 2.4G  标准差
        self.normalstd5 = 4   # 5G    标准差
        self.avr_ap = avr_ap  # 1 为均匀分布  其他为 随机分布

        " the speed"
        self.requireSNR = [2, 5, 9, 11, 15, 18, 20, 25, 29]
        self.rate = [6.5, 13, 19.5, 26, 39, 52, 58.5, 65]

        self.Pmax = 20         # dbm
        self.n = -84          # 噪声等级 （dBm）
        self.Cue = -80
        self.tao = 0.5
        self.packet_payload = 8184/1e6
        MACheader = 272
        PHYheader = 128
        ACK = 112 + PHYheader
        RTS = 160 + PHYheader
        CTS = 112 + PHYheader
        Bitrate = 1e6

        TACK = ACK / Bitrate
        TRTS = RTS / Bitrate
        TCTS = CTS / Bitrate

        PropagationDelay = 1e-6
        SlotTime = 50e-6
        SIFS = 28e-6
        DIFS = 128e-6


        self.Tsucc_p = TRTS + SIFS + TCTS + SIFS + (MACheader+PHYheader)/Bitrate + SIFS + TACK + DIFS
        self.Tidle = SlotTime
        self.Tcoll = RTS/Bitrate+DIFS
        self.APX, self.APY = self.Enviroment_AP()
        self.UEX, self.UEY = self.Enviroment_UE()
        self.LossAP2UE, self.LossAP2AP, self.LossUE2UE = self.loss_metrix()

    '''
    辅助函数
    '''
    def dB(self, a):
        return 10*np.log10(a)
    def idB(self,a):
        return 10**(a/10)

    '''
    环境配置部分
    '''
    def Enviroment_AP(self):
        if self.avr_ap == 1:
            APavenum = int(np.sqrt(self.NAP))
            avrlengH = self.MAZE_H / (APavenum + 1)
            avrlengW = self.MAZE_W / (APavenum + 1)
            APlocX = np.arange(0, self.MAZE_H, avrlengH)
            APlocY = np.arange(0, self.MAZE_W, avrlengW)
            APX=APlocX[1:]
            APY=APlocY[1:]
            outAPX = np.repeat(APX, APavenum)
            outAPY = np.zeros([self.NAP])
            # temp = np.repeat(APY, APavenum)
            # int()
            for loop1 in range(0, APavenum):
                temp = APY[np.arange(0-loop1, APavenum-loop1)]
                part = np.arange(0 + loop1 * APavenum, APavenum * (1 + loop1))
                for loop2 in range(0, APavenum):
                    outAPY[part[loop2]] = temp[loop2]
        else:
            outAPX = np.random.randint(1, self.MAZE_H, self.NAP)
            outAPY = np.random.randint(1, self.MAZE_W, self.NAP)

        return outAPX, outAPY

    def Enviroment_UE(self):
        UEX = np.random.randint(1, self.MAZE_H, self.NUE)
        UEY = np.random.randint(1, self.MAZE_W, self.NUE)
        return UEX, UEY

    def loss(self, UEX, UEY, APX, APY):
        distance = np.sqrt(pow(APX-UEX, 2)+pow(APY-UEY, 2))
        if self.freq == 2:
            shadefall = np.random.normal(0, self.normalstd2)
            Loss = self.P02 + 10*self.nsf2*np.log10(distance)+self.alpha2*distance+shadefall
        else:
            shadefall = np.random.normal(0, self.normalstd5)
            Loss = 10*self.nsf5*np.log10(distance)+self.alpha5*distance+shadefall
        return Loss

    def loss_metrix(self):
        UEX = self.UEX
        UEY = self.UEY
        APX = self.APX
        APY = self.APY
        # AP 2 UE
        LossAP2UE=np.zeros([self.NAP,self.NUE])
        for loop1 in range(self.NAP):
            for loop2 in range(self.NUE):
                LossAP2UE[loop1, loop2] = self.loss(UEX[loop2], UEY[loop2], APX[loop1], APY[loop1])
        # UE 2 UE
        LossUE2UE = np.zeros([self.NUE, self.NUE])
        for loop1 in range(self.NUE):
            for loop2 in range(self.NUE):
                LossUE2UE[loop1, loop2] = self.loss(UEX[loop2], UEY[loop2], UEX[loop1], UEY[loop1])
        # AP 2 AP
        LossAP2AP=np.zeros([self.NAP,self.NAP])
        for loop1 in range(self.NAP):
            for loop2 in range(self.NAP):
                LossAP2AP[loop1, loop2] = self.loss(APX[loop2], APY[loop2], APX[loop1], APY[loop1])
        LossAP2AP = (LossAP2AP+np.transpose(LossAP2AP))/2
        return LossAP2UE, LossAP2AP, LossUE2UE
    '''
    根据 输出的P 计算UE所连接的AP
    '''

    def connetion(self, P):
        LossAP2UE, LossAP2AP, LossUE2UE = self.LossAP2UE, self.LossAP2AP, self.LossUE2UE
        connetion = np.zeros([self.NUE])
        SNR = np.zeros([self.NUE])
        rate = np.zeros([self.NUE])
        re = np.zeros([self.NUE, self.NAP])
        for ue in range(0, self.NUE):
            record = np.array([])
            for fap in range(0, self.NAP):
                power = P[fap] - LossAP2UE[fap, ue] - self.n
                if power > self.Cue:
                    record = np.append(record, power)
                else:
                    record = np.append(record, -1e6)
            re[ue, :] = record
            loc = np.where(record>0)
            if loc == []:
                connetion[ue] = -1
                SNR[ue] = -100
                rate[ue] = 0
            else:
                connetion[ue] = np.argmax(record)
                SNR[ue] = np.max(record)
                findnear = np.argmin(abs(SNR[ue] - self.requireSNR))
                if SNR[ue]>np.max(self.requireSNR):
                    rate[ue] = np.max(self.requireSNR)
                elif SNR[ue] < np.min(self.requireSNR):
                    rate[ue] = 0
                elif SNR[ue] >= self.requireSNR[findnear]:
                    rate[ue] = self.rate[findnear]
                elif SNR[ue] < self.requireSNR[findnear]:
                    rate[ue] = self.rate[findnear - 1]
        return connetion, SNR, rate

    '''
    根据 P和C 计算 吞吐量
    '''
    def calculation_NP(self, P, C):
        '''
        只考虑下行信道，所以不考虑UE对AP的影响 但是这个地方要计算AP对UE的影响
        '''
        LossAP2UE, LossAP2AP, LossUE2UE = self.LossAP2UE, self.LossAP2AP, self.LossUE2UE
        # the first ord
        # calculation for AP
        totalAP = np.zeros([self.NAP, self.NAP])
        power_recordAP = np.zeros([self.NAP, self.NAP])
        for ap in range(0, self.NAP):
            for fap in range(0, self.NAP):
                power = self.dB(self.idB(P[fap]-LossAP2AP[fap, ap])+self.idB(self.n))
                power_recordAP[ap, fap] = power
                if power > C[ap]:
                    totalAP[ap, fap] = 1

            # 不考虑ue 对于 AP的影响
            # for fue in range(self.NUE, self.NAP + self.NUE):
            #     power = self.Pue - LossAP2UE[ap, fue] - self.n
            #     if power > C[ap]:
            #         totalAP[ap, fap] = 1
        for i in range(0, self.NAP):
            totalAP[i,i] = 0

        # calculation for UE
        totalUE = np.zeros([self.NUE, self.NAP])
        power_recordUE = np.zeros([self.NUE, self.NAP])
        for ue in range(0, self.NUE):
            for fap in range(0, self.NAP):  # type: int
                power = self.dB(self.idB(P[fap] - LossAP2UE[fap, ue]) + self.idB(self.n))
                if power > self.Cue:
                    totalUE[ue, fap] = 1
                    power_recordUE[ue, fap] = power
            # 不考虑UE影响
            # for fue in range(self.NUE, self.NAP + self.NUE):
            #     power = self.idB(self.Pue - LossUE2UE[ue, fue]) + self.idB(self.n)
            #     if self.dB(power) > self.Cue:
            #         totalUE[ap, fue] = 1
        # non interference set
        noAP = []
        oneAP = np.zeros([self.NAP])
        for ap in range(0, self.NAP):
            num = np.where(totalAP[ap, :] != 1)[0]
            noAP.append(num)
            oneAP[ap] = self.NAP - num.shape[0]
        noUE = []
        oneUE = np.zeros([self.NUE])
        for ue in range(0, self.NUE):
            num = np.where(totalUE[ue, :] != 1)[0]
            noUE.append(num)
            oneUE[ue] = self.NAP - num.shape[0]


        # the second order
        '''
        node1 node2 都不是AP的一阶节点 且 node1 和 node2 互相都不是
        '''
        twoAP = np.zeros([self.NAP])
        secordAP = []
        for ap in range(0, self.NAP):
            tempAP = []
            apset = set(noAP[ap])
            apset = apset - {ap}
            '选择node1'
            for node1 in apset:
                node1set = set(noAP[node1])
                node1set = node1set - {node1}
                node2set = apset & node1set
                for node2 in node2set:
                    power = self.dB( self.idB(P[node1] - LossAP2AP[node1, ap]) \
                            + self.idB(P[node2] - LossAP2AP[node2, ap]) \
                            + self.idB(self.n))
                    if power > C[ap]:
                        tempAP.append([node1, node2])
            secordAP.append(tempAP)
            twoAP[ap] = len(tempAP)/2

        twoUE = np.zeros([self.NUE])
        secordUE = []
        for ue in range(0, self.NUE):
            tempUE = []
            ueset = set(noUE[ue])
            '选择node1'
            for node1 in ueset:
                node1set = set(noAP[node1])
                node1set = node1set - {node1}
                node2set = ueset & node1set
                '选择node2'
                for node2 in node2set:
                    power = self.dB(self.idB(P[node1] - LossAP2UE[node1, ue]) \
                            + self.idB(P[node2] - LossAP2UE[node2, ue]) \
                            + self.idB(self.n))
                    if power > self.Cue:
                        tempUE.append([node1, node2])
                        # print('影响节点一:', P[node1] - LossAP2UE[node1, ue], node1,
                        #       '影响节点二:', P[node2] - LossAP2UE[node2, ue], node2,
                        #       '最终功率：', power)
            # print(tempUE)
            secordUE.append(tempUE)
            twoUE[ue] = len(tempUE)/2

        NumAP = twoAP + oneAP
        NumUE = twoUE + oneUE
        return NumAP, NumUE



    def through_out(self, P, C):
        connetion, SNR, rate = self.connetion(P)
        NumAP, NumUE = self.calculation_NP(P, C)
        thought_out = 0
        for i in range(self.NUE):
            if rate[i] != 0:
                con = int(connetion[i])
                nt = NumAP[con]
                nr = NumUE[i]
                n = nt + nr
                Pt = 1-(1-self.tao)**n
                Ps = self.tao*(1-self.tao)**(n-1)/Pt
                Pidle = 1-Pt
                Psucc = Pt*Ps
                Pcoll = Pt*(1-Ps)
                Tsucc = self.packet_payload/rate[i]
                thought_out += Psucc*self.packet_payload/\
                          (Psucc*Tsucc+Pidle*self.Tidle+Pcoll*self.Tcoll)
        return thought_out

    ########################################################
    #  画图部分
    ########################################################






if __name__ == '__main__':
    #%% calculpart
    env = scenario(NAP=9, NUE=100, freq=2, avr_ap=1)
    placeAPx,placeAPy = env.APX, env.APY
    placeUEx,placeUEy = env.UEX, env.UEY
    LossAP2UE, LossAP2AP, LossUE2UE = env.loss_metrix()
    connet, SNR, rate = env.connetion([20]*env.NAP)
    NumAP, NumUE = env.calculation_NP([20]*env.NAP, [-70]*env.NAP)
    th=env.through_out([20]*env.NAP, [-70]*env.NAP)



    #%%   test for loss and rate
    '''
    test for loss and rate
    '''
    loss = []
    rate = []
    for dis in range(0, 200):
        l=env.loss(0, 0, 0, dis)
        snr = 20-l+84
        loss.append(l)
        findnear = np.argmin(abs(snr - env.requireSNR))
        if snr < np.min(env.requireSNR):
            ra = 0
        elif snr > np.max(env.requireSNR):
            ra = 65
        elif snr >= env.requireSNR[findnear]:
            ra = env.rate[findnear]
        elif snr < env.requireSNR[findnear]:
            ra = env.rate[findnear-1]
        print(snr,ra, findnear)
        rate.append(ra)
    print(len(rate))
    plt.figure(1)
    plt.plot(list(range(0, 200)), loss)
    plt.show()
    plt.figure(2)
    plt.plot(list(range(0, 200)), rate)
    plt.show()

    '''
    test for throughout 
    '''

    #%%   test for UE SNR of UE

    plt.figure(1)
    ax = plt.gca()
    pap = ax.scatter(placeAPx, placeAPy, marker='v')
    n = 94
    for i in range(n,n+1):
        pue0 = ax.scatter(placeUEx[i], placeUEy[i])
        for j in range(env.NAP):
            ax.text((placeUEx[i]+placeAPx[j])/2, (placeUEy[i]+placeAPy[j])/2, str(int(LossAP2UE[j, i])), color='r')
        plt.show()
    #%%   test for link of all UE
    plt.figure(2)
    pue = plt.scatter(placeUEx, placeUEy)
    pap = plt.scatter(placeAPx, placeAPy, marker='v')
    for i in range(env.NAP):
        plt.text(placeAPx[i], placeAPy[i], str(i), color='k')
    for i in range(env.NUE):
        px = int(connet[i])
        plt.text(placeUEx[i],  placeUEy[i], str(i),color='r')
        plt.plot([placeUEx[i], placeAPx[px]], [placeUEy[i], placeAPy[px]], linestyle=":")
    plt.show()

    #%%   test for num right







