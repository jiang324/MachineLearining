import pandas as pd
import numpy as np
# LightGBM是轻量级（Light）的梯度提升机器（GBM）,是GBDT模型的另一个进化版本。
# 它延续了 XGBoost 的那一套集成学习的方式，相对于xgboost， 具有训练速度快和内存占用率低的特点
import lightgbm as lgb
import time
# sklearn.preprocessing.LabelEncoder()：标准化标签，将标签值统一转换成range(标签值个数-1)范围内
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold  # 交叉验证
from sklearn.metrics import mean_squared_error as mse  # MSE均方误差(求得结果与标签最大最小值对比)
# 用法：scipy.stats.linregress(x, y=None, alternative='two-sided')
# 计算两组测量的线性least-squares 回归。
# https://vimsky.com/examples/usage/python-scipy.stats.linregress.html
from scipy.stats import linregress


# 官方自定义评价函数
def get_score(y, y_pred, name):
    if name == 'CPU':
        t = 0.9 * np.abs(y - y_pred) / 100.
        return np.mean(t)
    else:
        max_v = np.max([y, y_pred], axis=0)
        max_v[max_v == 0.] = 1
        t = 0.1 * np.divide(np.abs(y - y_pred), max_v)
        return np.mean(t)


df_train = pd.read_csv('C:/Users/86178/Desktop/train.csv')
df_test = pd.read_csv('C:/Users/86178/Desktop/evaluation_public.csv')

cpu_feats = ['CU', 'STATUS', 'QUEUE_TYPE', 'PLATFORM', 'QUEUE_ID',
             'CPU_USAGE', 'MEM_USAGE',
             'LAUNCHING_JOB_NUMS', 'RUNNING_JOB_NUMS',
             'RESOURCE_TYPE', 'DISK_USAGE',
             'TIME_HOUR']


# 构造label shift平移构建目标值
def make_label(data):
    data['CPU_USAGE_1'] = data.CPU_USAGE.shift(-1)
    data['CPU_USAGE_2'] = data.CPU_USAGE.shift(-2)
    data['CPU_USAGE_3'] = data.CPU_USAGE.shift(-3)
    data['CPU_USAGE_4'] = data.CPU_USAGE.shift(-4)
    data['CPU_USAGE_5'] = data.CPU_USAGE.shift(-5)

    data['LAUNCHING_JOB_NUMS_1'] = data.LAUNCHING_JOB_NUMS.shift(-1)
    data['LAUNCHING_JOB_NUMS_2'] = data.LAUNCHING_JOB_NUMS.shift(-2)
    data['LAUNCHING_JOB_NUMS_3'] = data.LAUNCHING_JOB_NUMS.shift(-3)
    data['LAUNCHING_JOB_NUMS_4'] = data.LAUNCHING_JOB_NUMS.shift(-4)
    data['LAUNCHING_JOB_NUMS_5'] = data.LAUNCHING_JOB_NUMS.shift(-5)

    return data.dropna()  # 默认丢弃含有缺失值的行


# 处理时间
def proc_time(df):
    df['DOTTING_TIME'] /= 1000
    # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，
    # 并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
    # map函数要经过list转换，即：list(map(function,list1[]))
    df['DOTTING_TIME'] = list(map(
        lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)), df['DOTTING_TIME']))
    df = df.sort_values(['QUEUE_ID', 'DOTTING_TIME'])
    df['DOTTING_TIME'] = pd.to_datetime(df['DOTTING_TIME'])  # 将字符型的时间日期转换为时间型的数据
    df['TIME_HOUR'] = df['DOTTING_TIME'].map(lambda x: x.hour)
    return df


df_train = proc_time(df_train)
df_test = proc_time(df_test)

# 数值变换及交叉特征
# CPU_USAGE/MEM_USAGE等呈长尾分布，进行目标变换（这里使用开根号）使得序列更加平稳。
# 交叉特征
# CU与CPU_USAGE相乘，CU与MEM_USAGE相乘，将资源的百分比占用量转换为实际占用资源量
# MEMS_USAGE与DISK_USAGE相加，表征存储资源总占用
# 多种任务数量特征与CU相除，表征单位资源上的任务数量
for df in [df_train, df_test]:
    df['CPU_USAGE'] = 10 * np.sqrt(df['CPU_USAGE'])
    df['MEM_USAGE'] = 10 * np.sqrt(df['MEM_USAGE'])
    df['DISK_USAGE'] = 10 * np.sqrt(df['DISK_USAGE'])
    df['CU_CPU'] = df['CU'] * df['CPU_USAGE'] / 100.
    df['CU_MEM'] = df['CU'] * 4 * df['MEM_USAGE'] / 100.  # 1CU等于1核4GB
    df['TO_DO_JOB'] = df['LAUNCHING_JOB_NUMS'] - df['RUNNING_JOB_NUMS']
    df['MEM_DISK'] = df['MEM_USAGE'] + df['DISK_USAGE']

cpu_feats = cpu_feats + ['CU_CPU', 'CU_MEM', 'TO_DO_JOB', 'MEM_DISK']
# 结果：['CU', 'STATUS', 'QUEUE_TYPE', 'PLATFORM', 'QUEUE_ID',
#       'CPU_USAGE', 'MEM_USAGE', 'LAUNCHING_JOB_NUMS', 'RUNNING_JOB_NUMS',
#       'RESOURCE_TYPE', 'DISK_USAGE', 'TIME_HOUR', 'CU_CPU', 'CU_MEM', 'TO_DO_JOB', 'MEM_DISK']

# 时序特征
for name in ['CPU_USAGE', 'MEM_USAGE', 'CU_CPU', 'MEM_DISK']:
    f = [name]
    # 多个均值
    for n in range(1, 5):
        df_train[name + '_%d_ago' % n] = df_train[name].shift(n)
        df_test[name + '_%d_ago' % n] = df_test[name].shift(n)
        cpu_feats.append(name + '_%d_ago' % n)
        f.append(name + '_%d_ago' % n)

        df_train[name + '_mean_%d' % n] = df_train[f].mean(axis=1)
        df_test[name + '_mean_%d' % n] = df_test[f].mean(axis=1)
        cpu_feats.append(name + '_mean_%d' % n)
    # 趋势值   np.subtract():按元素减去参数
    df_train[name + '_0_trade'] = np.subtract(df_train[name], df_train[name + '_mean_4'])
    df_test[name + '_0_trade'] = np.subtract(df_test[name], df_test[name + '_mean_4'])
    cpu_feats.append(name + '_0_trade')
    for n in range(1, 5):
        df_train[name + '_%d_ago_trade' % n] = np.subtract(df_train[name + '_%d_ago' % n], df_train[name + '_mean_4'])
        df_test[name + '_%d_ago_trade' % n] = np.subtract(df_test[name + '_%d_ago' % n], df_test[name + '_mean_4'])
        cpu_feats.append(name + '_%d_ago_trade' % n)

for name in ['CPU_USAGE', 'CU_CPU']:
    for d in range(1, 4):
        df_train[name + '_mean_%d_ratio' % d] = np.divide(df_train[name + '_mean_%d' % d] + 1,
                                                          df_train[name + '_mean_4'] + 1)
        df_test[name + '_mean_%d_ratio' % d] = np.divide(df_test[name + '_mean_%d' % d] + 1,
                                                         df_test[name + '_mean_4'] + 1)

    cpu_feats = cpu_feats + [name + '_mean_1_ratio', name + '_mean_2_ratio', name + '_mean_3_ratio']

# job类时序特征
for name in ['RUNNING_JOB_NUMS', 'TO_DO_JOB']:
    f = [name]
    for n in range(1, 5):
        df_train[name + '_%d_ago' % n] = df_train[name].shift(n)
        df_test[name + '_%d_ago' % n] = df_test[name].shift(n)
        cpu_feats.append(name + '_%d_ago' % n)
        f.append(name + '_%d_ago' % n)

    df_train[name + '_mean'] = df_train[f].mean(axis=1)
    df_test[name + '_mean'] = df_test[f].mean(axis=1)
    cpu_feats.append(name + '_mean')

for name in ['RUNNING_JOB_NUMS', 'TO_DO_JOB']:
    df_train[name + '_0_trade'] = np.subtract(df_train[name], df_train[name + '_mean'])
    df_test[name + '_0_trade'] = np.subtract(df_test[name], df_test[name + '_mean'])
    if name == 'RUNNING_JOB_NUMS':  # or name == 'TO_DO_JOB':
        cpu_feats.append(name + '_0_trade')
    for n in range(1, 5):
        df_train[name + '_%d_ago_trade' % n] = np.subtract(df_train[name + '_%d_ago' % n], df_train[name + '_mean'])
        df_test[name + '_%d_ago_trade' % n] = np.subtract(df_test[name + '_%d_ago' % n], df_test[name + '_mean'])
        if name == 'RUNNING_JOB_NUMS' or name == 'TO_DO_JOB':
            cpu_feats.append(name + '_%d_ago_trade' % n)

print(df_train.shape)

# 差分  这个操作实际等效于：df - df.shift(1)
for name in ['CPU_USAGE', 'MEM_USAGE', 'CU_CPU', 'MEM_DISK']:
    df_train[name + '_diff'] = df_train[name].diff()
    df_test[name + '_diff'] = df_test[name].diff()
    cpu_feats.append(name + '_diff')
    for d in range(1, 4):
        df_train[name + '_diff_%d' % d] = df_train[name + '_diff'].shift(d)
        df_test[name + '_diff_%d' % d] = df_test[name + '_diff'].shift(d)
        cpu_feats.append(name + '_diff_%d' % d)

# 按队列名和时间聚合统计
for name in ['CPU_USAGE', 'MEM_USAGE', 'CU_CPU']:
    # groupby函数是先将df按照某个字段进行拆分，将相同属性分为一组；然后对拆分后的各组执行相应的转换操作；最后输出汇总转换后的各组结果
    tdf = df_train.groupby(['TIME_HOUR', 'QUEUE_ID'])[name].agg(
        {'mean', 'median', 'std', 'skew', 'max', 'min'}).reset_index()
    # 求均值，求中位数，求方差，求偏度
    # 偏度（skewness），是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。偏度(Skewness)亦称偏态、偏态系数。
    # 偏度是离群值(Outliers)导致的。离群值是那些正数中特别大或者负数中特别小的值，也就是绝对值特别大的值
    # 数据清洗时，会将带空值的行删除，此时DataFrame或Series类型的数据不再是连续的索引，可以使用reset_index()重置索引。
    # 在获得新的index，原来的index变成数据列，保留下来。不想保留原来的index，使用参数 drop=True，默认 False。
    tdf.rename(columns={
        'mean': name + '_T_QID_mean',
        'median': name + '_T_QID_median',
        'std': name + '_T_QID_std',
        'skew': name + '_T_QID_skew',
        'max': name + '_T_QID_max',
        'min': name + '_T_QID_min',
    }, inplace=True)  # 如果需要原地修改需要带上inplace=True的参数，否则原dataframe列名不会发生改变
    cpu_feats = cpu_feats + [name + x for x in [
        '_T_QID_mean', '_T_QID_median', '_T_QID_std', '_T_QID_skew',
        '_T_QID_max', '_T_QID_min']]
    # 根据一个或多个键将不同的DatFrame链接起来。
    df_train = pd.merge(df_train, tdf, on=['TIME_HOUR', 'QUEUE_ID'], how='left')
    df_test = pd.merge(df_test, tdf, on=['TIME_HOUR', 'QUEUE_ID'], how='left')

df_train = df_train.groupby('QUEUE_ID').apply(make_label).reset_index(drop=True)
# drop=True就是把原来的索引index列去掉，重置index

print(df_train.shape)


# 斜率
def lr(x1, x2, x3, x4, x5):
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([x1, x2, x3, x4, x5], dtype='float')
    return linregress(x, y)[0]


# linregress(x,y) 能够返回5个参数，分别是slope（斜率）, intercept（截距）, r_value（相关性）, p_value, stderr_slope

# 计算给出的5个点CPU_USAGE斜率
df_train['k'] = list(map(lambda x1, x2, x3, x4, x5: lr(x1, x2, x3, x4, x5),
                         df_train.CPU_USAGE_4_ago, df_train.CPU_USAGE_3_ago, df_train.CPU_USAGE_2_ago,
                         df_train.CPU_USAGE_1_ago,
                         df_train.CPU_USAGE))

df_test['k'] = list(map(lambda x1, x2, x3, x4, x5: lr(x1, x2, x3, x4, x5),
                        df_test.CPU_USAGE_4_ago, df_test.CPU_USAGE_3_ago, df_test.CPU_USAGE_2_ago,
                        df_test.CPU_USAGE_1_ago,
                        df_test.CPU_USAGE))
cpu_feats.append('k')

# 类别处理
# LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。其中包含以下常用方法：
# fit(y) ：fit可看做一本空字典，y可看作要塞到字典中的词。
# fit_transform(y)：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。
# inverse_transform(y)：根据索引值y获得原始数据。
# transform(y) ：将y转变成索引值
for name in ['STATUS', 'QUEUE_TYPE', 'PLATFORM', 'RESOURCE_TYPE', 'QUEUE_ID']:
    le = LabelEncoder()
    df_train[name] = le.fit_transform(df_train[name])
    df_test[name] = le.transform(df_test[name])
    df_train[name] = df_train[name].astype('category')
    # 转换为分类数据  可以指定特定的列转为分类数据 df['col1'] = df['col1'].astype('category')
    df_test[name] = df_test[name].astype('category')

print(df_train.shape)
print(df_test.shape)

targets_names = ['CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1',
                 'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2',
                 'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3',
                 'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4',
                 'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5']

df = pd.DataFrame()
# 去除特定列下面的重复行, 保留最后一次出现的数据
df_test = df_test.drop_duplicates(subset=['ID'], keep='last')
df['ID'] = df_test['ID']
print(df.shape)

# 直接利用规则给出job的预测
for name in targets_names:
    df[name] = df_test['LAUNCHING_JOB_NUMS']
score_all = []

for (i, name) in enumerate(targets_names):
    print('===================================================', name)
    if name.split('_')[0] == 'CPU':
        feats = cpu_feats.copy() + targets_names[:i:2]
    elif name.split('_')[0] == 'LAUNCHING':
        df_test[name] = df[name]
        continue
    else:
        continue
    print(feats)

    y = 0
    mse_score = []
    kfold = KFold(n_splits=4, shuffle=True, random_state=2222)
    # KFold（n_split, shuffle, random_state）
    # 　参数：n_splits:要划分的折数
    # 　　　　shuffle: 是否进行数据打乱
    # 　　　　random_state：如果shuffle是True，指定的种子值
    # split(a,b)：方法会根据折数对a和b进行划分。
    # 例如n_splits = 10，则划分为10折，其中9折在a中，1折在b中进行选择。最后返回的是相应数据的下标
    score = []
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[name])):
        print('--------------------------------------------------Fold ', fold_id)
        train = df_train.loc[trn_idx]   # loc[]函数用行列标签选择数据，前闭后闭
        train_x = train[feats]
        train_y = train[name]
        val = df_train.loc[val_idx]
        val_x = val[feats]
        val_y = val[name]
        print(train_x.shape)

        # 转换为Dataset数据格式
        train_matrix = lgb.Dataset(train_x, label=train_y)
        val_matrix = lgb.Dataset(val_x, label=val_y)
        params = {
            'boosting_type': 'gbdt',    # 设置提升类型  gbdt:传统梯度提升决策树
            'num_leaves': 20,   # 叶子节点数
            'objective': 'mse',  # 目标函数
            'learning_rate': 0.05,  # 学习速率
            'seed': 2,  # seed值，保证模型复现
            'verbose': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
            'nthread': -1,  # LightGBM 的线程数
        }       # https://www.jianshu.com/p/f36ceac676a7参数含义
        # 模型训练
        model = lgb.train(params, train_matrix, num_boost_round=10000,
                          valid_sets=[train_matrix, val_matrix], verbose_eval=10000, early_stopping_rounds=50)
        # model=clf.train(params,
        #                 train_set=train_matrix,  #训练样本
        #                 valid_sets=valid_matrix,  #测试样本
        #                 num_boost_round=10000,    #迭代次数，原来为2000
        #                 verbose_eval=100,#
        #                 early_stopping_rounds=500)  #如果数据在500次内没有提高，停止计算，原来为200
        # 模型预测
        y += model.predict(df_test[feats])
        pred_val = model.predict(val_x)
        mse_score.append(mse(pred_val, val_y))
        score.append(get_score((pred_val / 10) ** 2, (val_y / 10) ** 2, name.split('_')[0]))
        # 由于get_score函数计算时使用的是绝对值，因此顺序没关系
    print('mse_score: ', np.mean(mse_score))
    print('score: ', np.mean(score))
    score_all.append(np.mean(score))
    df[name] = y / 4
    df.loc[df[name] < 0, name] = 0
    if name.split('_')[0] == 'CPU':
        df.loc[df[name] > 100, name] = 100
    df_test[name] = df[name]

# 之前开根号，需要还原
f = ['CPU_USAGE_1', 'CPU_USAGE_2', 'CPU_USAGE_3', 'CPU_USAGE_4', 'CPU_USAGE_5']
for ff in f:
    df[ff] = (df[ff] / 10) ** 2

print(score_all)
print(1 - np.sum(score_all))

submit = pd.read_csv('C:/Users/86178/Desktop/submit_example.csv')[['ID']]
submit = pd.merge(submit, np.round(df).astype(int), on='ID')

# CPU后处理 整体扩大
targets_names = ['CPU_USAGE_1', 'CPU_USAGE_2', 'CPU_USAGE_3', 'CPU_USAGE_4', 'CPU_USAGE_5']
for i, name in enumerate(targets_names):
    idx1 = submit[name] >= 80
    idx2 = (submit[name] < 80) & (submit[name] >= 20)

    submit.loc[idx1, name] = submit.loc[idx1, name] * 1.05
    submit.loc[idx2, name] = submit.loc[idx2, name] * (1.1 + i * 0.01)
    submit.loc[submit[name] > 100, name] = 100

submit = np.round(submit).astype(int)

submit.to_csv('lgb_result.csv', index=False)
