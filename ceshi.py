import os
import pandas as pd
import tensorflow as tf
from deepmatch.models import *
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from preprocess import gen_data_set, gen_model_input
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler



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

# 3. 模型训练
if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()
else:
    K.set_learning_phase(True)


model = DSSM(user_feature_columns, item_feature_columns,user_dnn_hidden_units=(128,64, embedding_dim),
             item_dnn_hidden_units=(64, embedding_dim,),loss_type='softmax',sampler_config=sampler_config)


# Configures the model for training.
model.compile(optimizer="adam", loss=sampledsoftmaxloss)



history = model.fit(train_model_input, train_label,  # train_label,
                    batch_size=256, epochs=5, verbose=1, validation_split=0.0, ) # verbose是展示进度的几种方式
# validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集，0.0就是全部都是训练集


print(f"这是一个断点")
