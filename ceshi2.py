# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ceshi2
   Description :
   Author :       yzh
   date：          2022/10/20
-------------------------------------------------
   Change Activity:
                   2022/10/20:
-------------------------------------------------
"""
# __author__ = 'yzh'
# import numpy as np
# import sys
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from deepctr.feature_column import SparseFeat, VarLenSparseFeat,get_feature_names
# sys.path.append("..")
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] =  "7"
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from deepctr.models import DeepFM
#
# def split(x):
#     key_ans = x.split('|')
#     for key in key_ans:
#         if key not in key2index:
#             # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
#             key2index[key] = len(key2index) + 1
#     return list(map(lambda x: key2index[x], key_ans))
#
# data_path = "./"
#
# unames = ['user_id','gender','age','occupation','zip']
# user = pd.read_csv(data_path+'ml-1m/users.dat',sep='::',header=None,names=unames)
# rnames = ['user_id','movie_id','rating','timestamp']
# ratings = pd.read_csv(data_path+'ml-1m/ratings.dat',sep='::',header=None,names=rnames)
# mnames = ['movie_id','title','genres']
# movies = pd.read_csv(data_path+'ml-1m/movies.dat',sep='::',header=None,names=mnames,encoding="unicode_escape")
# movies['genres'] = list(map(lambda x: x.split('|')[0], movies['genres'].values))
#
# data = pd.merge(pd.merge(ratings,movies),user)#.iloc[:10000]  # 这里生成一个有十列的dataframe数据，也就是说这里的特征有十项
#
# sparse_features = ["movie_id", "user_id",
#                    "gender", "age", "occupation", "zip"] #准备特征
#
#
# target = ['rating']#准备标签
# #特征数值化 data里面3个特征都用LabelEncoder处理一下 就是把特征里面的值 从0到n开始编号
# for f in  sparse_features:
#     transfor = LabelEncoder()
#     data[f] = transfor.fit_transform(data[f])
#
# key2index = {}
# genres_list = list(map(split, data['genres'].values))
# genres_length = np.array(list(map(len, genres_list)))
# max_len = max(genres_length)
# # Notice : padding=`post`
# genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
#
# # 2.count #unique features for each sparse field and generate feature config for sequence feature
#
# fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4)
#                           for feat in sparse_features]
#
# use_weighted_sequence = False
# if use_weighted_sequence:
#     varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
#         key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
#                                                weight_name='genres_weight')]  # Notice : value 0 is for padding for sequence input feature
# else:
#     varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
#         key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
#                                                weight_name=None)]  # Notice : value 0 is for padding for sequence input feature
import os
import pandas as pd
import tensorflow as tf
from deepmatch.models import *
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from preprocess import gen_data_set, gen_model_input
from deepctr.feature_column import SparseFeat, VarLenSparseFeat,get_feature_names
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from Contrastive_Learning_Two_Tower.depends.deepctr.models.deepfm import DeepFM


os.environ["CUDA_VISIBLE_DEVICES"] =  "0"
SEQ_LEN = 50
negsample = 0

#1. 数据预处理(数据集-ml-1m)
#1.1 对rating评分进行处理的函数
def multiToSingleType(DataSet):
    DataSet_copy = DataSet.copy()
    rows_normal = DataSet_copy[DataSet_copy['rating'] >= 4]  # 获取标签小于1所在行的数据（得到一个dataframe）
    rows_abnormal = DataSet_copy[(DataSet_copy['rating'] > 0 ) & (DataSet_copy['rating'] < 4)]  # 获取标签大于0所在行的数据（得到一个dataframe）

    for i in range(rows_normal.shape[0]):
        DataSet_copy.loc[rows_normal.index[i], 'rating'] = 1  # 将标签小于1所在行的标签置为0
    for j in range(rows_abnormal.shape[0]):
        DataSet_copy.loc[rows_abnormal.index[j], 'rating'] = 0  # 将标签大于0所在行的标签置为1

    return DataSet_copy

# 对整个movielens进行调整的函数
def data_preprocess(datapath):
    data_path = str(datapath)
    unames = ['user_id','gender','age','occupation','zip']
    user = pd.read_csv(data_path+'/ml-1m/users.dat',sep='::',header=None,names=unames)
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(data_path + '/ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(data_path + '/ml-1m/movies.dat', sep='::', header=None, names=mnames, encoding="unicode_escape")
    ratings = multiToSingleType(ratings)
    return user,ratings,movies




user,ratings,movies = data_preprocess("/home/zhyang/My_Project/Contrastive_Learning_Two_Tower/depends/datasets")
movies['genres'] = list(map(lambda x: x.split('|')[0], movies['genres'].values)) # 对于每部电影的类型，只用第一种类型来计算
# 对所有有交互的数据整合到一起
data = pd.merge(pd.merge(ratings,movies),user)

sparse_features = ["movie_id", "user_id","gender", "age", "occupation", "zip", "genres"]
feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    # 对每个特征进行编码处理 data原本有十个特征，对十个特征中的七个进行编码创建
    data[feature] = lbe.fit_transform(data[feature]) + 1 # 这里是编码后每一列的值都加1
    # 寻找最长的列，padding对齐
    feature_max_idx[feature] = data[feature].max() + 1
user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
item_profile = data[["movie_id","genres"]].drop_duplicates('movie_id')
user_profile.set_index("user_id", inplace=True)
# 根据user_id进行分组，所以有多少个user就会有多少个组，然后将其看过的电影movie_id生成一个列表
user_item_list = data.groupby("user_id")['movie_id'].apply(list)

# 划分数据集
# 这里输入的data是有完整信息的data == data = pd.merge(pd.merge(ratings,movies),user)
train_set, test_set = gen_data_set(data, SEQ_LEN, negsample)
# 生成喂入模型的数据
train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

#2 开始构造模型输入
embedding_dim = 32
# feature_max_idx[feature]  == vocabulary_size------如果是one_hot编码的话位数大小应该是多少,也就是该项目总共有多少个？？？
# class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
#                                   ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):

user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                        SparseFeat("gender", feature_max_idx['gender'], 16),
                        SparseFeat("age", feature_max_idx['age'], 16),
                        SparseFeat("occupation", feature_max_idx['occupation'], 16),
                        SparseFeat("zip", feature_max_idx['zip'], 16),
                        VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                    embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                        VarLenSparseFeat(SparseFeat('hist_genres', feature_max_idx['genres'], embedding_dim,
                                                    embedding_name="genres"), SEQ_LEN, 'mean', 'hist_len'),
                        ]

item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim),
                        SparseFeat('genres', feature_max_idx['genres'], embedding_dim)
                        ]


train_counter = Counter(train_model_input['movie_id'])
item_count = [train_counter.get(i,0) for i in range(item_feature_columns[0].vocabulary_size)]
# 看一下item_count 到底找到了什么东西 ---就是每部电影被评价的次数
sampler_config = NegativeSampler('inbatch',num_sampled=255,item_name="movie_id",item_count=item_count) # num_sampled是正负样本对比 1正：255负，所以一个batch为256

linear_feature_columns = user_feature_columns + item_feature_columns
dnn_feature_columns = user_feature_columns + item_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
# model_input = {name: data[name] for name in sparse_features}  #
# model_input["genres"] = genres_list
# model_input["genres_weight"] = np.random.randn(data.shape[0], max_len, 1)

# 4.Define Model,compile and train
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_model_input, train_label,
                    batch_size=256, epochs=5, verbose=1, validation_split=0.2, )

print("misson completed")