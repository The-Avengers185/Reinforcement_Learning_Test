# 导入需要使用的库,其中numpy是支持数组和矩阵运算的科学计算库,而matplotlib是绘图库
import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)  # 找到获奖概率最大的拉杆索引
        self.best_prob = self.probs[self.best_idx] #  获奖概率最大的拉杆的获奖概率
        self.K = K  # 拉杆个数

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    """多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit  # 多臂老虎机实例
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆被拉动的次数
        self.regret = 0.0  # 累计遗憾值
        self.actions = []  # 记录每次选择的拉杆
        self.regrets = [] 

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError
    
    def run(self, num_steps):
        # 运行num_steps步,每一步都选择一根拉杆并更新懊悔值
        for _ in range(num_steps):
            k = self.run_one_step() # 选择一根拉杆
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

np.random.seed(1) # 设置随机种子,使实验可重复
# K = 10
# bandit_10_arm = BernoulliBandit(K) # 创建一个10臂老虎机实例
# print("随机生成了一个%d臂伯努利老虎机" % K)
# print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %(bandit_10_arm.best_idx, bandit_10_arm.best_prob))
