from collections import defaultdict
import copy
import numpy as np
import pandas

class KMEANS:
    def __init__(self, n_cluster, epsilon=1e-2, maxstep=2000):
        self.n_cluster = n_cluster
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.N = None
        self.centers = None
        self.cluster = defaultdict(list)

    def init_param(self, data):
        # 初始化参数, 包括初始化簇中心
        self.N = data.shape[0]
        random_ind = np.random.choice(self.N, size=self.n_cluster)
        self.centers = [data[i] for i in random_ind]  # list存储中心点坐标数组
        for ind, p in enumerate(data):
            self.cluster[self.mark(p)].append(ind)
        return

    def _cal_dist(self, center, p):
        # 计算点到簇中心的距离平方
        return sum([(i - j) ** 2 for i, j in zip(center, p)])

    def mark(self, p):
        # 计算样本点到每个簇中心的距离，选取最小的簇
        dists = []
        for center in self.centers:
            dists.append(self._cal_dist(center, p))
        return dists.index(min(dists))

    def update_center(self, data):
        # 更新簇的中心坐标
        for label, inds in self.cluster.items():
            self.centers[label] = np.mean(data[inds], axis=0)
        return

    def divide(self, data):
        # 重新对样本聚类
        tmp_cluster = copy.deepcopy(self.cluster)  # 迭代过程中，字典长度不能发生改变，故deepcopy
        for label, inds in tmp_cluster.items():
            for i in inds:
                new_label = self.mark(data[i])
                if new_label == label:  # 若类标记不变，跳过
                    continue
                else:
                    self.cluster[label].remove(i)
                    self.cluster[new_label].append(i)
        return

    def cal_err(self, data):
        # 计算MSE
        mse = 0
        for label, inds in self.cluster.items():
            partial_data = data[inds]
            for p in partial_data:
                mse += self._cal_dist(self.centers[label], p)
        return mse / self.N

    def fit(self, data):
        self.init_param(data)
        step = 0
        while step < self.maxstep:
            step += 1
            self.update_center(data)
            self.divide(data)
            err = self.cal_err(data)
            if err < self.epsilon:
                break
        return

# 再设计谱聚类算法
class Spectrum:
    def __init__(self, n_cluster, epsilon=1e-4, maxstep=1500, method='normalized_Lsym',
                 criterion='gaussian', gamma=2.0):
        self.n_cluster = n_cluster
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.method = method  # 本程序提供规范化以及非规范化的谱聚类算法
        self.criterion = criterion  # 相似性矩阵的构建方法
        self.gamma = gamma  # 高斯方法中的sigma参数


        self.W = None  # 图的相似性矩阵
        self.L = None  # 图的拉普拉斯矩阵
        self.L_norm = None  # 规范化后的拉普拉斯矩阵
        self.D = None  # 图的度矩阵
        self.cluster = None

        self.N = None

    def init_param(self, X):
        # 初始化参数
        self.N = X.shape[0]
        # dis_mat = self.cal_dis_mat(data)
        dis_mat = X
        self.cal_weight_mat(dis_mat)
        self.D = np.diag(self.W.sum(axis=1))
        self.L = self.D - self.W
        return

    def cal_weight_mat(self, dis_mat):
        # 计算相似性矩阵
        if self.criterion == 'gaussian':  # 适合于较小样本集
            if self.gamma is None:
                raise ValueError('gamma is not set')
            self.W = np.exp(-self.gamma * dis_mat)
        else:
            raise ValueError('the criterion is not supported')
        return

    def fit(self, X):
        # 训练主函数
        self.init_param(X)
        if self.method == 'normalized_Lsym':
            D = np.linalg.inv(np.sqrt(self.D))
            L = D @ self.L @ D
            w, v = np.linalg.eig(L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
            normalizer = np.linalg.norm(Vectors, axis=1)
            normalizer = np.repeat(np.transpose([normalizer]), self.n_cluster, axis=1)
            Vectors = Vectors / normalizer
        elif self.method == 'normalized_Lrw':#use the Lrw
            D = np.linalg.inv(self.D)
            L = np.identity(self.N) - D @ self.W
            w, v = np.linalg.eig(L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
        else:
            raise ValueError('the method is not supported')
        km = KMEANS(self.n_cluster, self.epsilon, self.maxstep)
        km.fit(Vectors)
        self.cluster = km.cluster
        return


if __name__ == '__main__':
    from itertools import cycle
    import matplotlib.pyplot as plt
    import csv

    num_cluster = 13 # 类别数

    # x = np.loadtxt(open("1F_matrix.csv","rb"),delimiter=",",skiprows=0)
    APname = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20,
              21, 23, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
              47, 48, 49, 50, 60, 61, 62, 63, 64, 65, 66, 67]
    I = pandas.read_csv("matrix_N7.csv")
    I = I.iloc[:, 1:]
    I = np.array(I.iloc[APname, APname])
    x = I
    x = (x.T + x)/2
    np.savetxt('1F_Inter_matrix.csv', x, delimiter=',')


    sp = Spectrum(n_cluster=num_cluster, method='normalized_Lsym', criterion='gaussian', gamma=2)
    sp.fit(x)
    cluster = sp.cluster
    print(cluster.items())
    listchannel = list(cluster.values())
    c =0
    for i in range((len(listchannel))):
        a = 0
        print(listchannel[i])
        for j in range(len(listchannel[i])):
            for k in range(len(listchannel[i])):
                a += x[listchannel[i][j]][listchannel[i][k]]
                b = a/2
        c += b
        print(b)
    print('C = ',c)




