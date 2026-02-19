import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to u.item file (MovieLens 100k dataset)
item_file = 'ml-100k/u.item'

# MovieLens u.item columns
columns = [
    'movie_id', 'title', 'release_date', 'video_release_date',
    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]

# Load dataset
movies = pd.read_csv(item_file, sep='|', names=columns, encoding='latin-1')

# Combine genres into a single string for TF-IDF
genre_cols = columns[5:]  # genre columns
movies['genres'] = movies[genre_cols].apply(lambda x: ' '.join([genre_cols[i] for i, val in enumerate(x) if val == 1]), axis=1)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movie(title, cosine_sim=cosine_sim):
    if title not in movies['title'].values:
        return f"Movie '{title}' not found in dataset."
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Test
movie_name = 'Toy Story (1995)'
recommendations = recommend_movie(movie_name)

print(f"Because you liked '{movie_name}':")
for i, movie in enumerate(recommendations, start=1):
    print(f"{i}. {movie}")