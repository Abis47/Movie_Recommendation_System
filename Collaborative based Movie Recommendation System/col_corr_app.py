import numpy as np
import pandas as pd

movies_df = pd.read_csv('movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
rating_df = pd.read_csv('ratings.csv',usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

df = pd.merge(rating_df, movies_df, on='movieId')

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

moviemat = df.pivot_table(index='userId',columns='title',values='rating')


movie_name = input("Enter Movie Name: ")

movie_user_ratings = moviemat[movie_name]

similar_to_movie = moviemat.corrwith(movie_user_ratings)

corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
corr_movie.dropna(inplace=True)

corr_movie = corr_movie.join(ratings['num of ratings'])
print(corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False).head())

