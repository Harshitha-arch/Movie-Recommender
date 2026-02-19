# CineMatch - Movie Recommendation System

A content-based movie recommender built with the MovieLens 100K dataset, scikit-learn, and a Streamlit frontend.

---

## How It Works

The system uses **TF-IDF vectorization** on movie genre tags combined with **cosine similarity** to find films with the closest genre profile to a given input.

**Pipeline:**

```
u.item (MovieLens) -> Genre string per movie -> TF-IDF matrix -> Cosine similarity matrix -> Top-N results
```

1. Each movie's binary genre flags (Action, Comedy, Drama, etc.) are joined into a genre string e.g. `"Animation Children Comedy"`
2. `TfidfVectorizer` converts these into weighted term vectors, down-weighting common genres like Drama
3. `cosine_similarity` computes pairwise similarity across all 1,682 movies
4. At query time, the input movie's row is extracted and the top-N most similar movies are returned (excluding itself)

> **Why TF-IDF over raw genre matching?** It penalises overly common genres, so a film tagged only `"Animation Children"` scores closer to Toy Story than a film tagged across 8 genres that happen to include Animation.

---

## Project Structure

```
Recommendation System/
├── app.py                  # Streamlit frontend
├── movie_recommender.py    # Standalone CLI script
├── ml-100k/
│   ├── u.item              # Movie metadata (used)
│   ├── u.data              # User ratings (available for extension)
│   └── ...
└── README.md
```

---

## Setup

```bash
# 1. Download dataset
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip

# 2. Install dependencies
pip install pandas scikit-learn streamlit

# 3. Run the web app
streamlit run app.py

# or run the CLI version
python movie_recommender.py
```

---

## Example Output

**Input:** `Toy Story (1995)`

```
Because you liked 'Toy Story (1995)':
1. Aladdin and the King of Thieves (1996)   -- Animation, Children, Comedy
2. Aristocats, The (1970)                   -- Animation, Children, Comedy
3. Pinocchio (1940)                         -- Animation, Children, Comedy
4. Sword in the Stone, The (1963)           -- Animation, Children, Comedy
5. Fox and the Hound, The (1981)            -- Animation, Children
```

---

## Tech Stack

| Component | Library |
|---|---|
| Data handling | pandas |
| Vectorization | scikit-learn TfidfVectorizer |
| Similarity | scikit-learn cosine_similarity |
| Frontend | Streamlit |
| Dataset | MovieLens 100K (grouplens.org) |
