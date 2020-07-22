import numpy as np
from collections import Counter
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
import copy
import pydot
from sklearn import tree

# 分类的原理参考这篇博客
# https://blog.csdn.net/bf02jgtrs00xktcx/article/details/82719765

class GBDT:

    def __init__(self,target,n_estimators,lr,max_depth):
        '''
        :param target: regression or classifier
        :param n_estimators: 树的棵树
        :param lr: 学习率
        :param max_depth:树的最大深度
        '''
        self.target = target
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.tree_list = []

    @staticmethod
    # sigmiod 函数
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    # 计算损失函数
    def calc_mse(residual):
        if len(residual) == 0:
            return 0
        return np.var(residual) * len(residual)

    @staticmethod
    # 切分分类数据
    def split_data(data,feat,val,data_type='classifier'):
        if data_type == 'classifier':
            arr1 = data[np.nonzero(data[:,feat] == val)]
            arr2 = data[np.nonzero(data[:,feat] != val)]
            arr1_index = np.nonzero(data[:,feat] == val)[0]
            arr2_index = np.nonzero(data[:,feat] != val)[0]
        else:
            arr1 = data[np.nonzero(data[:,feat].astype(float) < val)]
            arr2 = data[np.nonzero(data[:,feat].astype(float) >= val)]
            arr1_index = np.nonzero(data[:,feat].astype(float) < val)[0]
            arr2_index = np.nonzero(data[:,feat].astype(float) >= val)[0]
        return arr1,arr2,arr1_index,arr2_index

    @staticmethod
    # 连续变量的切分点处理
    def continuity_params_process(arr,feat):
        c = arr[:,feat].astype(float)
        c_sort = sorted(set(c))
        new_c = []
        for i in range(len(c_sort)-1):
            val = (c_sort[i] + c_sort[i+1]) / 2
            new_c.append(val)
        return new_c

    # 选择最好的切分点
    # 满足基尼系数减少最快的方向
    def select_split(self,data,Y,residual):
        min_gini = np.inf
        best_feat = None
        best_val = None
        left = None
        right = None
        left_Y = None
        right_Y = None
        left_residual = None
        right_residual = None
        data_type = 'continuity'
        for i in range(data.shape[1]-1):
            c_set = self.continuity_params_process(data, i)
            for val in c_set:
                arr1,arr2,arr1_index,arr2_index = self.split_data(data,i,val,data_type)
                arr1_residual = residual[arr1_index]
                arr2_residual = residual[arr2_index]
                g1 = self.calc_mse(arr1_residual)
                g2 = self.calc_mse(arr2_residual)
                # g = len(arr1) / len(data) * g1 + len(arr2) / len(data) * g2
                g = g1 + g2 # 获取剩下最小的均方误差
                if min_gini > g:
                    min_gini = g
                    best_feat = i
                    best_val = val
                    left = arr1
                    right = arr2
                    left_Y = Y[arr1_index]
                    right_Y = Y[arr2_index]
                    left_residual = residual[arr1_index]
                    right_residual = residual[arr2_index]
        return best_feat,best_val,left,right,left_Y,right_Y,left_residual,right_residual

    # 构建递归树
    def create_tree(self,data,Y,residual,n=0):
        tree = {}
        if len(set(residual)) <= 1: # 如果残差都一样则不分裂
            return self.calc_node(Y,data[:,-1])
        # 如果数据的特征一模一样，则无法进一步切分
        # 返回
        dd = data[:,:-1].tolist()
        ddd = list(map(tuple,dd))
        cc = Counter(ddd)
        if len(cc) == 1: # 如果特征都一样则不用分裂
            return self.calc_node(Y,data[:,-1])
        best_feat,best_val,left,right,left_Y,right_Y,left_residual,right_residual = self.select_split(data,Y,residual)
        n += 1
        if n >= self.max_depth:
            tree[(best_feat, best_val, 'left')] = self.calc_node(left_Y,left[:,-1])
            tree[(best_feat, best_val, 'right')] = self.calc_node(right_Y,right[:,-1])
        else:
            tree[(best_feat,best_val,'left')] = self.create_tree(left,left_Y,left_residual)
            tree[(best_feat,best_val,'right')] = self.create_tree(right,right_Y,right_residual)
        return tree

    def calc_node(self,Y,pred):
        # 计算节点的值
        if self.target.startswith('reg'):
            return np.mean(pred)
        return np.sum(Y - pred) / np.sum(pred * (1 - pred))


    # 构建gbdt回归树
    def create_gbdt(self,dataset):
        data = copy.copy(dataset)
        Y = dataset[:,-1]
        if self.target.startswith('reg'):
            base_score = np.mean(Y)
        else:
            base_score = np.log(np.sum(data[:,-1]) / np.sum(1 - data[:,-1]))
        self.tree_list.append(base_score)
        for i in range(self.n_estimators):
            for j in range(len(data)):
                data[j,-1] = self.predict(data[j,:-1]) # 预测
            residual = Y - data[:, -1] # 残差
            if self.target.startswith('reg'):
                data[:, -1] = residual # 如果是回归问题，那么预测值=残差了
            self.tree_list.append(self.create_tree(data,Y,residual))

    # 预测单颗树
    def predict_one(self,tree,X):
        if type(tree) != dict:
            return tree
        for key in tree:
            if X[key[0]] < key[1]:
                r = tree[(key[0],key[1],'left')]
            else:
                r = tree[(key[0], key[1], 'right')]
            return self.predict_one(r, X)

    # 预测
    def predict(self,X):
        result = self.tree_list[0]
        for tree in self.tree_list[1:]:
            result += self.lr * self.predict_one(tree,X)
        if self.target.startswith('reg'):
            return result
        return self.sigmoid(result)


if __name__ == '__main__':
    # GBDT分类
    data = np.array([[1,-5,0],
                     [2,5,0],
                     [3,-2,1],
                     [1,2,1],
                     [2,0,1],
                     [6,-6,1],
                     [7,5,1],
                     [6,-2,0],
                     [7,2,0]
                     ])

    data = data.astype(float)

    n_estimators = 5 #估计数
    lr = 0.1
    max_depth = 2
    mygbdt_tree = GBDT(target='classifier',n_estimators=n_estimators,lr=lr,max_depth=max_depth)
    mygbdt_tree.create_gbdt(data)
    print("create gbdt:",mygbdt_tree.predict(data[5,:-1]))

    gbdt = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=lr,max_depth=max_depth)
    gbdt.fit(data[:,:-1],data[:,-1])
    print("GBDT:",gbdt.predict_proba([data[5,:-1]]))


    # dot_data= tree.export_graphviz(gbdt.estimators_[0,0], out_file=None)
    # graph = pydot.graph_from_dot_data(dot_data)
    # graph[0].write_png('iris_simple.png')

    print("-------------regression-------------")
    data = np.array([[5,20,1.1],
                     [7,30,1.3],
                     [21,70,1.7],
                     [30,60,1.8]
                     ])
    mygbdt_tree = GBDT(target='regression',n_estimators=n_estimators,lr=lr,max_depth=max_depth)
    mygbdt_tree.create_gbdt(data)
    print("create gbdt:",mygbdt_tree.predict(data[0,:-1]))


    gbdt_reg = GradientBoostingRegressor(n_estimators=n_estimators,learning_rate=lr,max_depth=max_depth)
    gbdt_reg.fit(data[:,:-1],data[:,-1])
    print("GBDT:",gbdt_reg.predict([data[0,:-1]]))
