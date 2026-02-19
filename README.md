# CineMatch 🎬
A content-based movie recommender using the MovieLens 100K dataset.

## How It Works
Genre tags per movie are vectorized with **TF-IDF** and compared using **cosine similarity**.
TF-IDF down-weights common genres (like Drama) so niche tags carry more signal.

## Setup
```bash
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip && unzip ml-100k.zip
pip install pandas scikit-learn streamlit
streamlit run app.py
```

## Example
```
Input: Toy Story (1995)
1. Aladdin and the King of Thieves (1996)
2. Aristocats, The (1970)
3. Pinocchio (1940)
4. Sword in the Stone, The (1963)
5. Fox and the Hound, The (1981)
```

## Stack
`pandas` · `scikit-learn` · `streamlit` · MovieLens 100K
