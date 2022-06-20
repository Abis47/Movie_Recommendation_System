import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations

from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv('movies.csv',usecols=['movieId','title','genres'],dtype={'movieId': 'int32', 'title': 'str', 'genres':'str'})

tfv = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,10) for c in combinations(s.split('|'), r=i)))
tfv_matrix = tfv.fit_transform(movies_df['genres'])

cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)

indices = pd.Series(movies_df.index, index=movies_df.title).drop_duplicates()

def give_rec_cosine(title, cosine_sim=cosine_sim):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    cos_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies 
    cos_scores = sorted(cos_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    cos_scores = cos_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in cos_scores]

    # Top 10 most similar movies
    return movies_df['title'].iloc[movie_indices]


movie_name = input("Enter Movie Name: ")
print("Recommended Movies: \n", give_rec_cosine(movie_name))