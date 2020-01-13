# # 生成对称矩阵，对角线上元素均为0
# import numpy as np
# X = np.random.rand(5 ** 2).reshape(5, 5) # 创建一个5 * 5方阵
# X = np.triu(X) # 保留其上三角部分
# X += X.T - 2 * np.diag(X.diagonal()) # 将上三角部分”拷贝”到下三角部分
#
# # 写到AP_interference.xlsx
# from openpyxl import Workbook
# wb = Workbook()
# ws = wb.active
# for r in X.tolist():
#     ws.append(r)
# wb.save(r'./data/AP_interference.xlsx')


from collections import defaultdict
import copy

# 先设计Kmeans算法
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
    def __init__(self, n_cluster, epsilon=1e-3, maxstep=1000, method='unnormalized',
                 criterion='gaussian', gamma=2.0, dis_epsilon=70, k=5):
        self.n_cluster = n_cluster
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.method = method  # 本程序提供规范化以及非规范化的谱聚类算法
        self.criterion = criterion  # 相似性矩阵的构建方法
        self.gamma = gamma  # 高斯方法中的sigma参数
        self.dis_epsilon = dis_epsilon  # epsilon-近邻方法的参数
        self.k = k  # k近邻方法的参数

        self.W = None  # 图的相似性矩阵
        self.L = None  # 图的拉普拉斯矩阵
        self.L_norm = None  # 规范化后的拉普拉斯矩阵
        self.D = None  # 图的度矩阵
        self.cluster = None

        self.N = None

    # def init_param(self, data):
    def init_param(self, X):
        # 初始化参数
        # self.N = data.shape[0]
        self.N = X.shape[0]
        # dis_mat = self.cal_dis_mat(data)
        dis_mat = X
        self.cal_weight_mat(dis_mat)
        self.D = np.diag(self.W.sum(axis=1))
        self.L = self.D - self.W
        return

    def cal_dis_mat(self, data):
        # 计算距离平方的矩阵
        dis_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dis_mat[i, j] = (data[i] - data[j]) @ (data[i] - data[j])
                dis_mat[j, i] = dis_mat[i, j]
        return dis_mat

    def cal_weight_mat(self, dis_mat):
        # 计算相似性矩阵
        if self.criterion == 'gaussian':  # 适合于较小样本集
            if self.gamma is None:
                raise ValueError('gamma is not set')
            self.W = np.exp(-self.gamma * dis_mat)
        elif self.criterion == 'k_nearest':  # 适合于较大样本集
            if self.k is None or self.gamma is None:
                raise ValueError('k or gamma is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                inds = np.argpartition(dis_mat[i], self.k + 1)[:self.k + 1]  # 由于包括自身，所以+1
                tmp_w = np.exp(-self.gamma * dis_mat[i][inds])
                self.W[i][inds] = tmp_w
        elif self.criterion == 'eps_nearest':  # 适合于较大样本集
            if self.dis_epsilon is None:
                raise ValueError('epsilon is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                inds = np.where(dis_mat[i] < self.dis_epsilon)
                self.W[i][inds] = 1.0 / len(inds)
        else:
            raise ValueError('the criterion is not supported')
        return

    # def fit(self, data):
    def fit(self, X):
        # 训练主函数
        # self.init_param(data)
        self.init_param(X)
        if self.method == 'unnormalized':
            w, v = np.linalg.eig(self.L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
        elif self.method == 'normalized':
            D = np.linalg.inv(np.sqrt(self.D))
            L = D @ self.L @ D
            w, v = np.linalg.eig(L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
            normalizer = np.linalg.norm(Vectors, axis=1)
            normalizer = np.repeat(np.transpose([normalizer]), self.n_cluster, axis=1)
            Vectors = Vectors / normalizer
        else:
            raise ValueError('the method is not supported')
        km = KMEANS(self.n_cluster, self.epsilon, self.maxstep)
        km.fit(Vectors)
        self.cluster = km.cluster
        return


if __name__ == '__main__':
    from itertools import cycle
    import matplotlib.pyplot as plt

    num_cluster = 3  # 类别数

    # data, label = make_blobs(centers=num_cluster, n_features=2, cluster_std=1.2, n_samples=5, random_state=1)
    import numpy as np

    X = np.array(range(100)).reshape(10, 10)
    X = np.triu(X)
    X += X.T - 2 * np.diag(X.diagonal())

    sp = Spectrum(n_cluster=num_cluster, method='normalized', criterion='gaussian', gamma=0.1)
    # sp.fit(data)
    sp.fit(X)
    cluster = sp.cluster
    print(cluster.items())

    km = KMEANS(num_cluster)
    # km.fit(data)
    km.fit(X)
    cluster = km.cluster
    print(cluster.items())


    def visualize(data, spcluster, kmcluster):
        color = 'bgrymck'
        plt.subplot(121)
        for col, inds in zip(cycle(color), spcluster.values()):
            partial_data = data[inds]
            plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col)
        plt.title("Spectral Clustering")

        plt.subplot(122)
        for col, inds in zip(cycle(color), kmcluster.values()):
            partial_data = data[inds]
            plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col)
        plt.title("Kmeans Clustering")
        plt.show()
        return


    # visualize(data, sp.cluster, km.cluster)
    visualize(X, sp.cluster, km.cluster)