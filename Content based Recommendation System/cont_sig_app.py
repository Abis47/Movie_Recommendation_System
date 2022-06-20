import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations

from sklearn.metrics.pairwise import sigmoid_kernel


movies_df = pd.read_csv('movies.csv',usecols=['movieId','title','genres'],dtype={'movieId': 'int32', 'title': 'str', 'genres':'str'})

tfv = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,10) for c in combinations(s.split('|'), r=i)))
tfv_matrix = tfv.fit_transform(movies_df['genres'])

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

indices = pd.Series(movies_df.index, index=movies_df.title).drop_duplicates()

def give_rec_sig(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_df['title'].iloc[movie_indices]


movie_name = input("Enter Movie Name: ")
print("Recommended Movies: \n", give_rec_sig(movie_name))