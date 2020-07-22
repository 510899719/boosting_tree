import numpy as np
from collections import Counter,defaultdict
import copy


n_estimators =10#树的棵数
MAX_DEPTH = 2
LR = 0.3
min_child_weight = 0 # 最小叶子节点占比权重
base_score = 0.5


# 回归：G = ypred - y,H = 1
# 分类：G = ypred - y,H = ypred * (1 - ypred)

class XGBoostModel:
    def __init__(self,target,n_estimators,lr,max_depth,min_child_weight,reg_lambda,reg_alpha,base_score):
        '''
        :param target: reg if target is a regression else classify
        :param n_estimators: cart树的棵树
        :param lr: 学习率
        :param max_depth: 树的最大深度
        :param min_child_weight: 最小叶子节点占比权重
        :param reg_lambda: l2正则
        :param reg_alpha: l1正则
        '''
        self.target = target
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.tree_list = []
        self.gain_list = []
        if self.target.startswith('reg'):
            self.base_score = base_score
        else:
            self.base_score = np.log(base_score / (1 - base_score))

    def calc_G(self,pred,y):
        return np.sum(pred - y)

    def calc_H(self,pred):
        if self.target.startswith('reg'):
            return len(pred)
        return np.sum(pred * (1 - pred))

    @staticmethod
    # 切分分类数据
    def split_data(data,feat,val,data_type='classifier'):
        if data_type == 'classifier':
            arr1 = data[np.nonzero(data[:,feat] == val)]
            arr2 = data[np.nonzero(data[:,feat] != val)]
        else:
            arr1 = data[np.nonzero(data[:,feat].astype(float) < val)]
            arr2 = data[np.nonzero(data[:,feat].astype(float) >= val)]
        return arr1,arr2,np.nonzero(data[:,feat].astype(float) < val)[0],np.nonzero(data[:,feat].astype(float) >= val)[0]

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
    # 满足Gain最大且大于0才分裂
    def select_split(self,data,Y):
        max_gain = -1
        best_feat = None
        best_val = None
        left = None
        right = None
        left_y = None
        right_y = None
        g_left = None
        h_left = None
        g_right = None
        h_right = None
        data_type = 'continuity'
        for i in range(data.shape[1]-1):
            # c_set = set(data[:, i])
            c_set = self.continuity_params_process(data,i)
            for val in c_set:
                arr1,arr2,arr1_index,arr2_index = self.split_data(data,i,val,data_type)
                gain, G_left, H_left, G_right, H_right = self.calc_gain(arr1,Y[arr1_index],arr2,Y[arr2_index])
                if max_gain < gain and gain > 0 and self.min_child_weight <= min(H_left,H_right):
                    max_gain = gain
                    best_feat = i
                    best_val = val
                    left = arr1
                    right = arr2
                    left_y = Y[arr1_index]
                    right_y = Y[arr2_index]
                    g_left = G_left
                    h_left = H_left
                    g_right = G_right
                    h_right = H_right
        if best_feat is None:
            # g = np.sum(data[:,-1] - Y)
            g = self.calc_G(data[:,-1],Y)
            # h = len(data)
            h = self.calc_H(data[:,-1])
            return best_feat,best_val,left,right,left_y,right_y,g,h,g,h
        self.gain_list.append({best_feat:max_gain})
        return best_feat,best_val,left,right,left_y,right_y,g_left,h_left,g_right,h_right

    def calc_gain(self,left,left_y,right,right_y):
        # G_left = np.sum(left[:,-1] - left_y)
        G_left = self.calc_G(left[:,-1],left_y)
        # H_left = len(left)
        H_left = self.calc_H(left[:,-1])

        # G_right = np.sum(right[:,-1] - right_y)
        G_right = self.calc_G(right[:,-1],right_y)
        # H_right = len(right)
        H_right = self.calc_H(right[:,-1])


        Gain = (G_left ** 2 / (H_left + self.reg_lambda) + G_right ** 2 / (H_right+self.reg_lambda) -
                (G_left + G_right) ** 2/ (H_left + H_right + self.reg_lambda))/2 - self.reg_alpha
        return Gain,G_left,H_left,G_right,H_right

    # 构建递归树
    def create_tree(self,data,Y,n=0):
        '''
        利用递归构建回归树，n用来限制树的最大深度
        '''
        tree = {}
        dd = data[:,:-1].tolist()
        ddd = list(map(tuple,dd))
        cc = Counter(ddd)
        if len(cc) == 1:
            g = self.calc_G(data[:,-1],Y)
            h = self.calc_H(data[:,-1])
            return -g / (h + self.reg_lambda)
        best_feat,best_val,left,right,left_y,right_y,g_left,h_left,g_right,h_right = self.select_split(data,Y)
        if best_feat is None:
            return -g_left / (h_left + self.reg_lambda)
        n += 1
        if n >= self.max_depth:
            tree[(best_feat,best_val,'left')] = -g_left / (h_left + self.reg_lambda)
            tree[(best_feat,best_val,'right')] = -g_right / (h_right + self.reg_lambda)
        else:
            tree[(best_feat,best_val,'left')] = self.create_tree(left,left_y,n)
            tree[(best_feat,best_val,'right')] = self.create_tree(right,right_y,n)
        return tree

    def fit(self,dataset):
        data = copy.copy(dataset)
        self.tree_list.append(self.base_score)
        for i in range(self.n_estimators):
            for j in range(len(data)):
                data[j,-1] = self.predict(data[j,:-1])
            self.tree_list.append(self.create_tree(data,dataset[:,-1]))

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
        return 1 / (1 + np.exp(-result))

    # 计算特征重要度
    def feat_importance(self):
        feat_imp = defaultdict(float)
        feat_counts = defaultdict(int)
        for item in self.gain_list:
            k, v = list(item.items())[0]
            feat_imp[k] += v
            feat_counts[k] += 1
        # 计算平均增益
        for k in feat_imp:
            feat_imp[k] /= feat_counts[k]
        v_sum = sum(feat_imp.values())
        for k in feat_imp:
            feat_imp[k] /= v_sum
        return feat_imp


from xgboost.sklearn import XGBRegressor,XGBClassifier

# 回归例子
data = np.array([[5,20,1.1],
                 [7,30,1.3],
                 [21,70,1.7],
                 [30,60,1.8],
                 [26,40,1.6],
                 ])

xgb = XGBRegressor(n_estimators=n_estimators,learning_rate=LR,max_depth=MAX_DEPTH,
                   min_child_weight=min_child_weight,base_score=base_score)
xgb.fit(data[:,:-1],data[:,-1])
print("xgboost:",xgb.predict(data[0,:-1].reshape(1,-1)))

my_xgb_tree = XGBoostModel(target='regression',n_estimators=n_estimators,lr=LR,max_depth=MAX_DEPTH,
                                min_child_weight=min_child_weight,reg_lambda=1,reg_alpha=0,base_score=base_score)
my_xgb_tree.fit(data)
print("my xgb tree:",my_xgb_tree.predict(data[0,:-1]))

print(xgb.feature_importances_)
print(my_xgb_tree.feat_importance())


print('----------------classify test---------------------')
data = np.array([[1,-5,0],
                 [2,5,0],
                 [3,-2,1],
                 [2,2,1],
                 [2,0,1],
                 [6,-6,1],
                 [7,5,1],
                 [6,-2,0],
                 [7,2,0]
                 ])
data = data.astype(float)

xgb = XGBClassifier(n_estimators=n_estimators,learning_rate=LR,max_depth=MAX_DEPTH,
                   min_child_weight=min_child_weight,base_score=base_score)
xgb.fit(data[:,:-1],data[:,-1])
print("xgboost:",xgb.predict_proba(data[0,:-1].reshape(1,-1)))

my_xgb_tree = XGBoostModel(target='classify',n_estimators=n_estimators,lr=LR,max_depth=MAX_DEPTH,
                                min_child_weight=min_child_weight,reg_lambda=1,reg_alpha=0,base_score=base_score)
my_xgb_tree.fit(data)
print("my xgb tree:",my_xgb_tree.predict(data[0,:-1]))

print('xgboost feature importance',xgb.feature_importances_)
print(my_xgb_tree.feat_importance())


