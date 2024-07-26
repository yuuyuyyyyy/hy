'''
import pandas as pd
path = 'D://recommondsystem//ez_douban'
movies = pd.read_csv('D://recommondsystem//ez_douban//movies.csv')
print('电影数目（有名称）：%d' % movies[~pd.isnull(movies.title)].shape[0])
print('电影数目（没有名称）：%d' % movies[pd.isnull(movies.title)].shape[0])
print('电影数目（总计）：%d' % movies.shape[0])
movies.sample(20)
ratings = pd.read_csv('D://recommondsystem//ez_douban//ratings.csv')
print('用户数据：%d' % ratings.userId.unique().shape[0])
print('评分数目：%d' % ratings.shape[0])
ratings.sample(20)
links = pd.read_csv('D://recommondsystem//ez_douban//links.csv')
links.sample(20)
'''

'''
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# 加载MovieLens数据集
movies = pd.read_csv('D://recommondsystem//ez_douban//movies.csv')
ratings = pd.read_csv('D://recommondsystem//ez_douban//ratings.csv')
links = pd.read_csv('D://recommondsystem//ez_douban//links.csv')

# 合并数据集
movie_data = pd.merge(ratings, movies, on='movieId')
movie_data = pd.merge(movie_data, links, on='movieId')

# 创建用户-电影评分矩阵
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# 将稀疏矩阵转换为压缩稀疏行矩阵（CSR）格式，以节省内存
user_movie_matrix_csr = csr_matrix(user_movie_matrix.values)

# 计算用户相似度矩阵
user_similarity = cosine_similarity(user_movie_matrix_csr)


# 定义推荐函数
def recommend_movies(user_id, top_n=5):
    user_index = user_id - 1
    similar_users = np.argsort(user_similarity[user_index])[::-1][1:]  # 获取与目标用户相似度从高到低的用户索引
    recommended_movies = []

    for user in similar_users:
        rated_movies = user_movie_matrix.iloc[user, :]
        unrated_movies = rated_movies[rated_movies == 0].index  # 获取未评分的电影
        recommended_movies.extend(unrated_movies)

    recommended_movies = list(set(recommended_movies))[:top_n]  # 获取前top_n个推荐电影

    return recommended_movies


# 测试推荐系统
user_id = 1
recommended_movies = recommend_movies(user_id, top_n=5)
print("为用户{}推荐的电影：".format(user_id))
for movie_title in recommended_movies:
    print(movie_title)
'''

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# 加载数据集
movies = pd.read_csv('D://recommondsystem//ez_douban//movies.csv')
ratings = pd.read_csv('D://recommondsystem//ez_douban//ratings.csv')
links = pd.read_csv('D://recommondsystem//ez_douban//links.csv')

# 合并数据集
movie_data = pd.merge(ratings, movies, on='movieId')
movie_data = pd.merge(movie_data, links, on='movieId')

# 创建用户-电影评分矩阵
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# 将稀疏矩阵转换为压缩稀疏行矩阵格式，以节省内存
user_movie_matrix_csr = csr_matrix(user_movie_matrix.values)

# 计算用户相似度矩阵
user_similarity = cosine_similarity(user_movie_matrix_csr)


# 定义推荐函数
def recommend_movies(user_id, top_n=5):
    user_index = user_id - 1
    similar_users = np.argsort(user_similarity[user_index])[::-1][1:]  # 获取与目标用户相似度从高到低的用户索引
    recommended_movies = []

    for user in similar_users:
        rated_movies = user_movie_matrix.iloc[user, :]
        unrated_movies = rated_movies[rated_movies == 0].index  # 获取未评分的电影
        recommended_movies.extend(unrated_movies)

    recommended_movies = list(set(recommended_movies))[:top_n]  # 获取前top_n个推荐电影

    return recommended_movies


# 构建推荐模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(user_movie_matrix.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(user_movie_matrix.shape[1], activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 训练模型
X = user_movie_matrix.values
y = user_movie_matrix.values
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# 测试推荐系统
user_id = 1
recommended_movies = recommend_movies(user_id, top_n=5)
print("为用户{}推荐的电影：".format(user_id))
for movie_title in recommended_movies:
    print(movie_title)
