import matplotlib
import pandas as pd
import numpy as np
import multiprocessing
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
import time

level_columns = ['user_id', 'level_id', 'count', 'duration_avg', 'success_count', 'win_duration_avg',
                 'fail_duration_avg', 'win_reststep_avg', 'fail_reststep_avg', 'retry_times', 'help_avg', 'date_times',
                 'first_time', 'last_time']
user_columns = ['user_id', 'count', 'duration_avg', 'success_count', 'win_duration_avg', 'fail_duration_avg',
                'win_reststep_avg', 'fail_reststep_avg', 'retry_times_avg', 'help_avg', 'date_covered', 'mini_level',
                'max_level', 'start_level',
                'last_level', 'level_num', 'last_level_pass', 'uf_avg_duration', 'uf_avg_passrate',
                'uf_avg_win_duration', 'uf_avg_retrytimes']
date_list = ['2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04']


def level_analyze(data):
    user_id = data[0][0]
    level_id = data[0][1]
    print("handling user_id = {}, level_id = {}".format(user_id, level_id))
    df = data[1].sort_values(by=['time'], ignore_index=True)
    count = len(df['user_id'])
    duration_avg = np.average(df['f_duration'])
    success_df = df[df['f_success'] == 1]
    fail_df = df[df['f_success'] == 0]
    success_count = success_df.shape[0]
    win_duration = np.sum(success_df['f_duration'])
    win_duration_avg = 0 if success_count == 0 else win_duration / success_count
    fail_duration = np.sum(fail_df['f_duration'])
    fail_duration_avg = 0 if count == success_count else fail_duration / (count - success_count)
    win_reststep_avg = 0 if success_count == 0 else np.sum(success_df['f_reststep']) / success_count
    fail_reststep_avg = 0 if count == success_count else np.sum(fail_df['f_reststep']) / (count - success_count)
    retry_times = count - 1
    help_avg = np.average(df['f_help'])
    date_times = '|'.join(df['time'])
    first_time = df['time'][0]
    last_time = df['time'][count - 1]
    return [user_id, level_id, count, duration_avg, success_count,
            win_duration_avg, fail_duration_avg, win_reststep_avg, fail_reststep_avg, retry_times, help_avg, date_times,
            first_time, last_time]


def user_analyze(data):
    # user_id: 用户id
    user_id = data[0]
    print("handling user_id = {}".format(user_id))
    df = data[1]
    # count: 闯关记录数
    count = np.sum(df['count'])
    # duration_avg: 平均耗时
    duration = np.sum(df['count'] * df['duration_avg'])
    duration_avg = 0 if count == 0 else duration / count
    # success_count: 通关次数
    success_count = np.sum(df['success_count'])
    # win_duration_avg: 通关平均耗时
    win_duration = np.sum(df['success_count'] * df['win_duration_avg'])
    win_duration_avg = 0 if success_count == 0 else win_duration / success_count
    # fail_duration_avg: 未通关平均耗时
    fail_duration_avg = 0 if count == success_count else (duration - win_duration) / (count - success_count)
    # win_reststep_avg: 通关剩余步数比例平均值
    win_reststep_avg = 0 if success_count == 0 else np.sum(df['success_count'] * df['win_reststep_avg']) / success_count
    # fail_reststep_avg: 未通关剩余步数比例平均值
    fail_reststep_avg = 0 if count == success_count else np.sum(
        df['fail_reststep_avg'] * (df['count'] - df['success_count'])) / (count - success_count)
    # retry_times_avg: 平均重试次数
    retry_times_avg = np.average(df['retry_times'])
    # help_avg: 辅助平均使用次数
    help_avg = np.average(df['help_avg'])
    # date_covered: 涵盖的日期（2/1 - 2/4）
    date_covered = 0
    all_times = '|'.join(df['date_times'])
    for date in date_list:
        date_covered += (1 if all_times.find(date) > -1 else 0)
    # mini_level: 最小关卡
    mini_level = np.min(df['level_id'])
    # max_level: 最大关卡
    max_level = np.max(df['level_id'])
    # start_level: 起始关卡
    start_level = df.sort_values(by=['first_time'], ascending=True).iloc[0].at['level_id']
    # last_level: 最终关卡
    last_record = df.sort_values(by=['last_time'], ascending=False).iloc[0]
    last_level = last_record.at['level_id']
    # level_num: 经过的关卡数
    level_num = df.shape[0]
    # last_level_pass: 最终关卡是否通过
    last_level_pass = 1 if last_record.at['success_count'] > 0 else 0
    # 关卡属性
    # uf_avg_duration：用户经过关卡，平均每次尝试花费的时间
    uf_avg_duration = np.average(df['f_avg_duration'])
    # uf_avg_passrate：用户经过关卡，平均通关率
    uf_avg_passrate = np.average(df['f_avg_passrate'])
    # uf_avg_win_duration：用户经过关卡，平均每次通关花费的时间
    uf_avg_win_duration = np.average(df['f_avg_win_duration'])
    # uf_avg_retrytimes： 用户经过关卡，平均重试次数
    uf_avg_retrytimes = np.average(df['f_avg_retrytimes'])
    return [user_id, count, duration_avg, success_count, win_duration_avg, fail_duration_avg, win_reststep_avg,
            fail_reststep_avg, retry_times_avg, help_avg, date_covered, mini_level, max_level, start_level, last_level,
            level_num, last_level_pass, uf_avg_duration, uf_avg_passrate, uf_avg_win_duration, uf_avg_retrytimes]


def construct_clf(clf_name):
    clf = None
    if clf_name == 'SVM':
        clf = svm.LinearSVC(dual=False)
    elif clf_name == 'DTree':
        clf = DecisionTreeClassifier()
    elif clf_name == 'NB':
        clf = BernoulliNB()
    clf = CalibratedClassifierCV(clf, cv=2, method='sigmoid')  # 概率校正
    return clf


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 隐藏层线性输出
        self.hidden = nn.Linear(n_feature, n_hidden)
        # self.hidden1 = nn.Linear(n_feature, n_hidden)
        # self.hidden2 = nn.Linear(n_hidden, n_hidden)
        # self.hidden3 = nn.Linear(n_hidden, n_hidden)
        # 输出层线性输出
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 前向传播
        x = torch.softmax(self.hidden(x), dim=1)  # 激励函数(隐藏层的非线性值)
        # x = torch.tanh(self.hidden1(x))
        # x = torch.sigmoid(self.hidden(x))
        # x = torch.relu(self.hidden(x))
        # x = torch.sigmoid(self.hidden1(x))  # 激励函数(隐藏层的非线性值)
        # x = torch.softmax(self.hidden2(x), dim=1)  # 激励函数(隐藏层的非线性值)
        # x = torch.relu(self.hidden3(x))  # 激励函数(隐藏层的非线性值)
        x = self.out(x)  # 输出值（预测值另算）
        return x


if __name__ == '__main__':
    # print(torch.__version__)
    # print('gpu', torch.cuda.is_available())
    train_df = pd.read_csv('./data/train.csv', sep='\t')
    dev_df = pd.read_csv('./data/dev.csv', sep='\t')
    test_df = pd.read_csv('./data/test.csv', sep='\t')
    user_analyze_df = pd.read_csv('./user_analyze.csv')
    user_test_analyze_df = pd.read_csv('./user_test_analyze.csv')

    # 移除相关性低的特征
    user_columns.remove('help_avg')
    user_columns.remove('mini_level')
    user_columns.remove('start_level')

    train_feat_df = train_df.merge(user_analyze_df, how='left', on='user_id')
    x_train = train_feat_df[user_columns]
    y_train = train_feat_df['label']
    dev_feat_df = dev_df.merge(user_analyze_df, how='left', on='user_id')
    x_dev = dev_feat_df[user_columns]
    y_dev = dev_feat_df['label']
    test_feat_df = test_df.merge(user_test_analyze_df, how='left', on='user_id')
    x_test = test_feat_df[user_columns]
    x_all = pd.concat([x_train, x_dev, x_test])
    x_all.set_index('user_id', inplace=True)
    # 归一化数据
    x_all = x_all.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    user_columns.remove('user_id')
    x_train = train_df.merge(x_all, how='left', on='user_id')[user_columns]
    x_dev = dev_df.merge(x_all, how='left', on='user_id')[user_columns]
    x_test = test_df.merge(x_all, how='left', on='user_id')[user_columns]

    # 建立RNN神经网络
    iter_num = 1000
    n_feature = len(user_columns)
    net = Net(n_feature=n_feature, n_hidden=5, n_output=2)
    print('net:\n', net)
    # 训练网络
    x_train_tensor = torch.from_numpy(x_train.values.astype('float32'))
    y_train_tensor = torch.from_numpy(y_train.values)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    loss_func = nn.CrossEntropyLoss()  # 误差形式是1D Tensor，预测值是2D Tensor (batch, n_classes)
    loss_arr = []
    step = 100
    for i in range(iter_num):
        out = net(x_train_tensor)
        loss = loss_func(out, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % step == 0:
            prediction = torch.softmax(out, dim=1).detach().numpy()
            print("Iteration: {}, Loss: {}, Train AUC: {}".format(i, loss.item(), roc_auc_score(y_train, prediction[:, 1])))
            loss_arr.append(loss.item())

    # 展现损失函数趋势
    # plt.figure(figsize=(15, 5))
    # plt.plot(range(0, iter_num, step), loss_arr, color='blue')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('RNN Loss Curve')
    # plt.show()

    # 验证集上使用模型，用于调参
    x_dev_tensor = torch.from_numpy(x_dev.values.astype('float32'))
    y_dev_predict = torch.softmax(net(x_dev_tensor), dim=1).detach().numpy()
    print('AUC on dev set = {}'.format(roc_auc_score(y_dev, y_dev_predict[:, 1])))

    # 生成结果文件
    x_test_tensor = torch.from_numpy(x_test.values.astype('float32'))
    y_predict = torch.softmax(net(x_test_tensor), dim=1).detach().numpy()
    result_df = pd.DataFrame()
    result_df['ID'] = test_df['user_id'].values
    result_df['Prediction'] = y_predict[:, 1]
    result_df.to_csv('./result-{}.csv'.format(round(time.time())), index=False)

    # train_df = pd.read_csv('./data/train.csv', sep='\t')
    # dev_df = pd.read_csv('./data/dev.csv', sep='\t')
    # test_df = pd.read_csv('./data/test.csv', sep='\t')
    # user_analyze_df = pd.read_csv('./user_analyze.csv')
    # user_test_analyze_df = pd.read_csv('./user_test_analyze.csv')
    # # 移除相关性低的特征+
    # user_columns.remove('user_id')
    # user_columns.remove('help_avg')
    # user_columns.remove('mini_level')
    # user_columns.remove('start_level')
    # # fit & predict
    # train_feat_df = train_df.merge(user_analyze_df, how='left', on='user_id')
    # x_train = train_feat_df[user_columns]
    # y_train = train_feat_df['label']
    # dev_feat_df = dev_df.merge(user_analyze_df, how='left', on='user_id')
    # x_dev = dev_feat_df[user_columns]
    # y_dev = dev_feat_df['label']
    # # using SVM
    # clf_svm = construct_clf('SVM')
    # clf_svm.fit(x_train, y_train)
    # print('accuracy using SVM = {}'.format(accuracy_score(y_dev, clf_svm.predict(x_dev))))
    # print('auc using SVM = {}'.format(roc_auc_score(y_dev, clf_svm.predict_proba(x_dev)[:, 1])))
    # # using Decision Tree
    # clf_dt = construct_clf('DTree')
    # clf_dt.fit(x_train, y_train)
    # print('accuracy using Decision Tree = {}'.format(accuracy_score(y_dev, clf_dt.predict(x_dev))))
    # print('auc using Decision Tree = {}'.format(roc_auc_score(y_dev, clf_dt.predict_proba(x_dev)[:, 1])))
    # # using NB
    # clf_nb = construct_clf('NB')
    # clf_nb.fit(x_train, y_train)
    # print('accuracy using NB = {}'.format(accuracy_score(y_dev, clf_nb.predict(x_dev))))
    # print('auc using NB = {}'.format(roc_auc_score(y_dev, clf_nb.predict_proba(x_dev)[:, 1])))
    #
    # # 生成结果文件
    # test_feat_df = test_df.merge(user_test_analyze_df, how='left', on='user_id')
    # x_test = test_feat_df[user_columns]
    # y_predict = clf_svm.predict_proba(x_test)[:, 1]
    # result_df = pd.DataFrame()
    # result_df['ID'] = test_df['user_id'].values
    # result_df['Prediction'] = y_predict
    # result_df.to_csv('./result.csv', index=False)

    # pool = multiprocessing.Pool()
    # train_df = pd.read_csv('./data/train.csv', sep='\t')
    # dev_df = pd.read_csv('./data/dev.csv', sep='\t')
    # test_df = pd.read_csv('./data/test.csv', sep='\t')
    # seq_df = pd.read_csv('./data/level_seq.csv', sep='\t')
    # meta_df = pd.read_csv('./data/level_meta.csv', sep='\t')
    # all_train_df = pd.concat([train_df, dev_df], ignore_index=True)
    #
    # # 数据清洗
    # # 1. 没用步数却通关，视为脏数据（观察user_id=10963发现）
    # dirty_percent = seq_df[(seq_df['f_reststep'] == 1.0) & (seq_df['f_success'] == 1)].shape[0] / \
    #                 seq_df[(seq_df['f_reststep'] == 1.0)].shape[0]
    # print("1. 没用步数却通关，视为脏数据，占'没用步数'数据比例 = {}".format(dirty_percent))
    # # 2. 同一用户在同一时间玩同一关卡，视为脏数据（观察user_id=2775发现）
    # ex = seq_df.groupby(['user_id', 'time', 'level_id']).agg({"user_id": ["count"]})
    # duplicated_num = ex.loc[ex[('user_id', 'count')] > 1][('user_id', 'count')].sum()
    # print("2. 同一用户在同一时间玩同一关卡，视为脏数据，占比 = {}".format(duplicated_num / seq_df.shape[0]))
    # # 3. 含有未通关但f_reststep并不为0记录（不符合题设要求的），经分析需要保留不做为脏数据
    # fail_but_reststep_df = pd.merge(
    #     seq_df[(seq_df['f_reststep'] != 0.0) & (seq_df['f_success'] == 0)]['user_id'].drop_duplicates(),
    #     all_train_df, how='left', on='user_id').dropna()
    # fail_but_reststep_percent = len(fail_but_reststep_df[fail_but_reststep_df['label'] == 1]) / \
    #                             fail_but_reststep_df.shape[0]
    # print("3. 含有未通关但f_reststep并不为0记录（不符合题设要求的）的用户，其中的留存比率 = {}".format(fail_but_reststep_percent))
    # # 去除脏数据
    # seq_df.drop(seq_df.index[(seq_df['f_reststep'] == 1.0) & (seq_df['f_success'] == 1)], inplace=True)
    # seq_df.drop_duplicates(subset=['user_id', 'time', 'level_id'], inplace=True)
    # print("数据清洗完成！")
    #
    # # 取样例数据分析
    # examples = seq_df[(seq_df['user_id'] == 2775) | (seq_df['user_id'] == 2776)]
    # grouped = examples.groupby(['user_id', 'level_id'])
    # res = pool.map(level_analyze, grouped)
    # exdf = pd.DataFrame(res, columns=level_columns)
    # exdf = exdf.merge(meta_df, how='left', on='level_id')
    # exdf.to_csv('./level_analyze.csv')
    # # 对比用户闯关特征数据同关卡属性，并没有发现某个关卡对用户而言，难度系数如何
    # # 但仍基于以上用户闯关特征提取该游戏对于用户而言的特征
    # all_train_seq_df = all_train_df.merge(seq_df, how='left', on='user_id')
    # user_level_grouped = all_train_seq_df.groupby(['user_id', 'level_id'])
    # user_level_analyze_list = pool.map(level_analyze, user_level_grouped)
    # # user_level_analyze = user_level_grouped.apply(level_analyze)
    # # user_level_analyze_df = pd.DataFrame(list(user_level_analyze), columns=level_columns)
    # user_level_analyze_df = pd.DataFrame(user_level_analyze_list, columns=level_columns)
    # user_level_analyze_df = user_level_analyze_df.merge(meta_df, how='left', on='level_id')
    # user_level_analyze_df.to_csv('./user_level_analyze.csv', index=False)
    # # user_id: 用户id
    # # count: 闯关记录数
    # # duration_avg: 平均耗时
    # # success_count: 通关次数
    # # win_duration_avg: 通关平均耗时
    # # fail_duration_avg: 未通关平均耗时
    # # win_reststep_avg: 通关剩余步数比例平均值
    # # fail_reststep_avg: 未通关剩余步数比例平均值
    # # retry_times_avg: 平均重试次数
    # # help_avg: 辅助平均使用次数
    # # date_covered: 涵盖的日期（2/1 - 2/4）
    # # mini_level: 最小关卡
    # # max_level: 最大关卡
    # # start_level: 起始关卡
    # # last_level: 最终关卡
    # # level_num: 经过的关卡数
    # # last_level_pass: 最终关卡是否通过
    # # 关卡属性
    # # uf_avg_duration：用户经过关卡，平均每次尝试花费的时间
    # # uf_avg_passrate：用户经过关卡，平均通关率
    # # uf_avg_win_duration：用户经过关卡，平均每次通关花费的时间
    # # uf_avg_retrytimes： 用户经过关卡，平均重试次数
    # user_grouped = user_level_analyze_df.groupby(['user_id'])
    # user_analyze_list = pool.map(user_analyze, user_grouped)
    # user_analyze_df = pd.DataFrame(user_analyze_list, columns=user_columns)
    # user_analyze_df.to_csv('./user_analyze.csv', index=False)
    #
    # # 针对测试集
    # test_seq_df = test_df.merge(seq_df, how='left', on='user_id')
    # user_level_test_grouped = test_seq_df.groupby(['user_id', 'level_id'])
    # user_level_test_analyze_list = pool.map(level_analyze, user_level_test_grouped)
    # user_level_test_analyze_df = pd.DataFrame(user_level_test_analyze_list, columns=level_columns)
    # user_level_test_analyze_df = user_level_test_analyze_df.merge(meta_df, how='left', on='level_id')
    # user_level_test_analyze_df.to_csv('./user_level_test_analyze.csv', index=False)
    #
    # user_test_grouped = user_level_test_analyze_df.groupby(['user_id'])
    # user_test_analyze_list = pool.map(user_analyze, user_test_grouped)
    # user_test_analyze_df = pd.DataFrame(user_test_analyze_list, columns=user_columns)
    # user_test_analyze_df.to_csv('./user_test_analyze.csv', index=False)
    #
    # # 试图找出变量和公式，同label关联，但找公式是机器学习干的事儿。在可用方法一定的情况下，关键在于找出变量！！！
    # print("")
    # pool.close()
