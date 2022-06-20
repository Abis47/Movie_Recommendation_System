import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

movies_df = pd.read_csv('movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
rating_df = pd.read_csv('ratings.csv',usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

df = pd.merge(rating_df, movies_df, on='movieId')

movie_ratingCount = (df.groupby(by = ['title'])['rating'].count().reset_index().rename(columns = {'rating': 'totalRatingCount'}))
rating_with_totalRatingCount = df.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')

popularity_threshold = 50
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

movie_pivot_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)

movie_pivot_df_matrix = csr_matrix(movie_pivot_df.values)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_pivot_df_matrix)

movie_name = input("Enter Movie Name: ")

query_index = list(movie_pivot_df.index).index(movie_name)

distances, indices = model_knn.kneighbors(movie_pivot_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_pivot_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_pivot_df.index[indices.flatten()[i]], distances.flatten()[i]))