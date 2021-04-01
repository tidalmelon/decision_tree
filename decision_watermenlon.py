# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from dtree_plot import createPlot


def peek_data(dataset):
    """
    查看下数据
    """
    print('head------')
    print(dataset.head(10))
    #   编号  色泽  根蒂  敲声  纹理  脐部  触感 好瓜
    #0   1  青绿  蜷缩  浊响  清晰  凹陷  硬滑  是
    #1   2  乌黑  蜷缩  沉闷  清晰  凹陷  硬滑  是
    #2   3  乌黑  蜷缩  浊响  清晰  凹陷  硬滑  是
    #3   4  青绿  蜷缩  沉闷  清晰  凹陷  硬滑  是
    #4   5  浅白  蜷缩  浊响  清晰  凹陷  硬滑  是
    #5   6  青绿  稍蜷  浊响  清晰  稍凹  软粘  是
    #6   7  乌黑  稍蜷  浊响  稍糊  稍凹  软粘  是
    #7   8  乌黑  稍蜷  浊响  清晰  稍凹  硬滑  是
    #8   9  乌黑  稍蜷  沉闷  稍糊  稍凹  硬滑  否
    #9  10  青绿  硬挺  清脆  清晰  平坦  软粘  否
    print('info------')
    print(dataset.info())
    #<class 'pandas.core.frame.DataFrame'>
    #RangeIndex: 17 entries, 0 to 16
    #Data columns (total 8 columns):
    # #   Column  Non-Null Count  Dtype
    #---  ------  --------------  -----
    # 0   编号      17 non-null     int64
    # 1   色泽      17 non-null     object
    # 2   根蒂      17 non-null     object
    # 3   敲声      17 non-null     object
    # 4   纹理      17 non-null     object
    # 5   脐部      17 non-null     object
    # 6   触感      17 non-null     object
    # 7   好瓜      17 non-null     object
    #dtypes: int64(1), object(7)
    #memory usage: 1.2+ KB
    #None

    print('value_count, 色泽------')
    print(dataset['色泽'].value_counts())
    #青绿    6
    #乌黑    6
    #浅白    5
    #Name: 色泽, dtype: int64
    print('value_count, 好瓜------')
    dd = dataset['好瓜'].value_counts()
    print(type(dd))
    print('Count:', dd.count())
    print('dd[0:2]------------')
    print(type(dd[0:1]), dd[0:1])
    print(type(dd[0:2]), dd[0:2])
    #否    9
    #是    8
    #Name: 好瓜, dtype: int64

    print('list(dataset)------')
    print(list(dataset))
    print(dataset.columns.values)
    print(dataset.columns.tolist())
    #['编号', '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    #['编号' '色泽' '根蒂' '敲声' '纹理' '脐部' '触感' '好瓜']
    #['编号', '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']


# 1. dataset: 训练集 [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],]

def entropy(total, sub_set, label):
    """
    信息熵接口
    """
    ent = 0.0
    for k, v in sub_set[label].value_counts().items():
        p = float(v)/total
        # 熵=信息量的数学期望
        ent += (p * (-np.log2(p)))
    return ent


def conditioal_entropy(dataset, attr, label):
    """
    条件熵接口
    """
    size = dataset.index.size

    ent_con = 0.0
    for attr_val, num in dataset[attr].value_counts().items():
        sub_set = dataset[dataset[attr]==attr_val]
        ent = entropy(num, sub_set, label)
        w = float(num)/size
        # 条件熵=sum(分桶的熵*权重)
        ent_con += w*ent
    return ent_con


def IG(dataset, attr, label):
    """
    信息增益接口
    """
    size = dataset.index.size
    H_D = entropy(size, dataset, label)
    H_D_A = conditioal_entropy(dataset, attr, label)
    # 信息增益=未分桶前熵-条件熵（分桶后）
    info_gain = H_D - H_D_A
    return info_gain


def get_attr_values(dataset):
    """
    获取所有{属性:[属性值列表]}
    全局使用
    """
    attri_list = dataset.columns.tolist()
    attri_list = attri_list[1:-1]
    dic_attr_valarr = {}
    for attr in attri_list:
        dic_attr_valarr[attr] = list(set(dataset[attr].values))
    return dic_attr_valarr


def optimal_attr(dataset, label):
    """
    最优属性划分
    """

    attri_list = dataset.columns.tolist()
    attri_list = attri_list[1:-1]

    ig_mx = float('-inf')
    attr_mx = None
    idx_mx = -1
    for idx, attr in enumerate(attri_list):
        ig = IG(dataset, attr, label)
        if ig > ig_mx:
            ig_mx = ig
            attr_mx = attr
            idx_mx = idx
    return idx_mx, attr_mx, ig_mx


def tree_generate(dataset, dic_attr_valarr, label):
    """
    构造决策树
    """

    labelval_num_list = dataset[label].value_counts()
    # 决策树退出条件1：当前数据集样本权属于同一类别：全是好瓜
    if labelval_num_list.size == 1:
        return labelval_num_list.index[0]

    # 副本，不干扰主数据集
    d = dataset.drop(columns=['编号', label])
    # 决策树退出条件2：属性集(attr)为空 或 所有样本在所有属性上取值相同，无法继续划分（都一样，但依然有好瓜，有坏瓜, 无法划分, 这个时候就取 先验概率）
    if d.columns.size == 0 or d.drop_duplicates().index.size == 1:
        return labelval_num_list.idxmax()

    idx_attr, attr, ig_mx = optimal_attr(dataset, label)
    print('最优划分属性:', idx_attr, attr, ig_mx)

    node = {attr: {}}
    v_c = dataset[attr].value_counts()
    for attr_val in dic_attr_valarr.get(attr):

        # 决策树退出条件3： 当前节点的样本集为空，不能划分(取先验概率最大的标签值：是, 否)
        if not v_c.get(attr_val):
            node[attr][attr_val] = labelval_num_list.idxmax()
        else:
            dataset_D_V = dataset[dataset[attr]==attr_val]
            dataset_D_V = dataset_D_V.drop(attr, axis=1)
            node[attr][attr_val] = tree_generate(dataset_D_V, dic_attr_valarr, label)

    return node 


def load_dataset(fname):
    """
    加载数据
    """
    return pd.read_csv(fname)


if __name__ == '__main__':

    fname = './watermelon2.0.csv'
    label = '好瓜'
    dataset = load_dataset(fname)
    dic_attr_valarr = get_attr_values(dataset)

    dtree = tree_generate(dataset, dic_attr_valarr, label)

    import json
    print(json.dumps(dtree, ensure_ascii=False))

    
    createPlot(dtree)
