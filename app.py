import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Page config 
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="centered",
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e0d5;
}

.stApp {
    background: #0a0a0f;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}

.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #c8a96e;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3rem, 8vw, 5.5rem);
    font-weight: 900;
    line-height: 1;
    color: #f5efe8;
    margin: 0;
    letter-spacing: -0.02em;
}

.hero-title span {
    color: #c8a96e;
    font-style: italic;
}

.hero-sub {
    font-size: 1rem;
    color: #7a7068;
    margin-top: 1rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Divider ── */
.gold-line {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c8a96e, transparent);
    margin: 1.5rem auto;
}

/* ── Search box ── */
.stTextInput > div > div > input {
    background: #13121a !important;
    border: 1px solid #2a2530 !important;
    border-radius: 4px !important;
    color: #e8e0d5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.85rem 1.2rem !important;
    transition: border-color 0.2s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: #c8a96e !important;
    box-shadow: 0 0 0 1px #c8a96e22 !important;
}

.stTextInput > div > div > input::placeholder {
    color: #3d3840 !important;
}

/* ── Button ── */
.stButton > button {
    background: #c8a96e !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
}

.stButton > button:hover {
    background: #d4b87a !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #c8a96e;
    margin: 2.5rem 0 1.2rem;
    text-align: center;
}

.movie-card {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    background: #13121a;
    border: 1px solid #1e1c25;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s ease, transform 0.15s ease;
}

.movie-card:hover {
    border-color: #c8a96e44;
    transform: translateX(4px);
}

.rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: #2a2530;
    min-width: 2rem;
    text-align: center;
    line-height: 1;
}

.movie-info {
    flex: 1;
}

.movie-title-card {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: #e8e0d5;
    margin: 0;
}

.movie-genres-card {
    font-size: 0.75rem;
    color: #5a5460;
    margin-top: 0.25rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

.star {
    color: #c8a96e;
    font-size: 0.8rem;
}

/* ── Error ── */
.error-box {
    background: #1a1015;
    border: 1px solid #3d1f28;
    border-left: 3px solid #c85a5a;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    color: #c87070;
    font-size: 0.9rem;
    margin-top: 1rem;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #13121a !important;
    border: 1px solid #2a2530 !important;
    border-radius: 4px !important;
    color: #e8e0d5 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-size: 0.8rem !important;
    color: #5a5460 !important;
    letter-spacing: 0.08em !important;
}

/* ── Footer ── */
.cinefooter {
    text-align: center;
    padding: 3rem 0 1rem;
    font-size: 0.72rem;
    color: #2e2b35;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# Data loading
@st.cache_data
def load_data():
    item_file = 'ml-100k/u.item'
    columns = [
        'movie_id', 'title', 'release_date', 'video_release_date',
        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    movies = pd.read_csv(item_file, sep='|', names=columns, encoding='latin-1')
    genre_cols = columns[5:]
    movies['genres'] = movies[genre_cols].apply(
        lambda x: ' '.join([genre_cols[i] for i, val in enumerate(x) if val == 1]), axis=1
    )
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return movies, cosine_sim


def recommend_movie(title, movies, cosine_sim, n=5):
    if title not in movies['title'].values:
        return None
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]


# Load
try:
    movies, cosine_sim = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# Hero
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ AI-Powered Discovery ✦</div>
    <h1 class="hero-title">Cine<span>Match</span></h1>
    <p class="hero-sub">Find your next favourite film</p>
</div>
<div class="gold-line"></div>
""", unsafe_allow_html=True)


if not data_loaded:
    st.markdown("""
    <div class="error-box">
        Dataset not found. Make sure <code>ml-100k/u.item</code> exists in the same folder.<br><br>
        Run: <code>curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip && unzip ml-100k.zip</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# Search mode toggle
col1, col2 = st.columns([2, 1])
with col1:
    search_mode = st.radio(
        "Search by", ["Type a title", "Browse the list"],
        horizontal=True,
        label_visibility="collapsed"
    )

with col2:
    n_results = st.selectbox("Results", [5, 10, 15], label_visibility="collapsed")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

movie_name = None

if search_mode == "Type a title":
    movie_name = st.text_input(
        "movie", 
        placeholder="e.g. Toy Story (1995), Fargo (1996)…",
        label_visibility="collapsed"
    )
    search_clicked = st.button("Find Similar Films →")
else:
    all_titles = sorted(movies['title'].tolist())
    movie_name = st.selectbox("Pick a movie", all_titles, label_visibility="collapsed")
    search_clicked = st.button("Find Similar Films →")


# Results
if search_clicked and movie_name:
    results = recommend_movie(movie_name.strip(), movies, cosine_sim, n=n_results)

    if results is None:
        st.markdown(f"""
        <div class="error-box">
            Could not find <strong>"{movie_name}"</strong> in the dataset.<br>
            Try the <em>Browse the list</em> mode to find the exact title.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-label">Because you watched "{movie_name}"</div>
        """, unsafe_allow_html=True)

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            genres_display = row['genres'].replace(' ', ' · ') if row['genres'] else 'Unknown'
            stars = "★" * min(rank, 1) if rank == 1 else ""
            st.markdown(f"""
            <div class="movie-card">
                <div class="rank">0{rank}</div>
                <div class="movie-info">
                    <div class="movie-title-card">{row['title']}</div>
                    <div class="movie-genres-card">{genres_display}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="cinefooter">
    Powered by MovieLens 100K · TF-IDF · Cosine Similarity
</div>
""", unsafe_allow_html=True)