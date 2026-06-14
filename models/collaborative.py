# =============================================================================
# models/collaborative.py
# Purpose: Collaborative Filtering using User-Item Matrix + Cosine Similarity
#
# THEORY:
# -------
# Collaborative Filtering (CF) recommends items by finding USERS with similar
# taste and suggesting what THEY liked. The core assumption is:
#   "Users who agreed in the past will agree in the future."
#
# Two variants exist:
#   1. User-Based CF : find similar USERS → recommend what they rated highly
#   2. Item-Based CF : find similar ITEMS based on who rated them → recommend
#
# This implementation uses ITEM-BASED CF (more scalable, more stable):
#   a) Build User-Item Matrix  (rows=users, cols=movies, values=ratings)
#   b) Compute Item-Item Cosine Similarity (transpose and compare columns)
#   c) For a given movie, find N most similar movies by rating patterns
#
# KEY DIFFERENCE from Content-Based:
#   - Content-Based uses movie FEATURES (genres)
#   - Collaborative uses USER BEHAVIOUR (ratings)
#   → CF can surface surprising recommendations (serendipity)
#   → CF suffers from cold-start: new movies with 0 ratings can't be recommended
#
# Sparsity problem: Most users only rate a few movies, so the matrix is very
# sparse. We handle this by only keeping movies with enough ratings.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class CollaborativeFilteringRecommender:
    """
    Item-Based Collaborative Filtering recommender.
    
    Attributes:
      movies         : movies DataFrame
      ratings        : ratings DataFrame
      user_item_matrix: pivot table (users × movies) of ratings
      item_similarity : cosine similarity between movies (based on rating vectors)
      movie_mapper    : movieId → column index in matrix
      movie_id_to_title: movieId → clean_title lookup
    """

    def __init__(self, min_ratings: int = 20):
        """
        Args:
          min_ratings (int): minimum number of ratings a movie must have
                             to be included (filters out very obscure movies).
                             Lower = more movies but sparser matrix.
        """
        self.min_ratings = min_ratings
        self.movies = None
        self.ratings = None
        self.user_item_matrix = None
        self.item_similarity = None
        self.movie_mapper = None
        self.movie_id_to_title = None
        self.title_to_movie_id = None

    # ------------------------------------------------------------------
    # Step 1: Build the User-Item matrix and compute item similarity
    # ------------------------------------------------------------------
    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        """
        Train the collaborative filtering model.
        
        Steps:
          a) Filter movies with too few ratings (reduces sparsity noise)
          b) Pivot ratings into a User × Movie matrix
          c) Fill missing ratings with 0 (user hasn't rated = 0)
          d) Compute Item × Item cosine similarity from rating patterns
        
        Args:
          movies  (pd.DataFrame): cleaned movies DataFrame
          ratings (pd.DataFrame): cleaned ratings DataFrame
        """
        self.movies = movies
        self.ratings = ratings

        # a) Build title lookup dictionaries
        self.movie_id_to_title = dict(
            zip(movies["movieId"], movies["clean_title"])
        )
        self.title_to_movie_id = {
            v.lower(): k for k, v in self.movie_id_to_title.items()
        }

        # b) Filter: keep only movies with >= min_ratings ratings
        movie_rating_counts = ratings.groupby("movieId").size()
        popular_movie_ids = movie_rating_counts[
            movie_rating_counts >= self.min_ratings
        ].index
        filtered_ratings = ratings[ratings["movieId"].isin(popular_movie_ids)]

        print(f"🔧 Movies with ≥{self.min_ratings} ratings: {len(popular_movie_ids):,}")

        # c) Pivot: rows=userId, cols=movieId, values=rating
        #    Missing entries (user never rated the movie) → 0
        print("🔧 Building User-Item matrix...")
        self.user_item_matrix = filtered_ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            aggfunc="mean"      # average if same user rated same movie twice
        ).fillna(0)

        print(f"   Matrix shape: {self.user_item_matrix.shape}")
        # → (n_users × n_popular_movies)

        # d) Compute Item-Item cosine similarity
        #    Transpose so rows = movies, each movie is a vector over users
        #    cos_sim[i][j] = how similarly users rated movie i vs movie j
        print("🔧 Computing Item-Item cosine similarity...")
        item_matrix = csr_matrix(self.user_item_matrix.T.values)
        self.item_similarity = cosine_similarity(item_matrix)

        # Map movieId to matrix column index
        self.movie_mapper = {
            mid: idx for idx, mid in enumerate(self.user_item_matrix.columns)
        }
        print(f"   Item similarity matrix shape: {self.item_similarity.shape}")
        print("✅ Collaborative Filtering model ready!\n")

    # ------------------------------------------------------------------
    # Step 2: Recommend movies similar to a given title
    # ------------------------------------------------------------------
    def recommend(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return the top-N movies most similar to the given title
        based on collaborative rating patterns.
        
        Args:
          movie_title (str): movie title to base recommendations on
          top_n       (int): number of recommendations to return
        
        Returns:
          pd.DataFrame with columns: rank, title, similarity_score
        
        Raises:
          ValueError: if movie not found or has too few ratings
        """
        title_lower = movie_title.lower().strip()

        # Fuzzy match against known titles
        matches = [t for t in self.title_to_movie_id if title_lower in t]
        if not matches:
            raise ValueError(
                f"❌ Movie '{movie_title}' not found in collaborative model. "
                "It may have fewer than the required number of ratings."
            )

        best_match_title = matches[0]
        movie_id = self.title_to_movie_id[best_match_title]

        # Check if movie_id is in our filtered matrix
        if movie_id not in self.movie_mapper:
            raise ValueError(
                f"❌ '{movie_title}' has fewer than {self.min_ratings} ratings "
                "and was excluded from the collaborative model. "
                "Try Content-Based filtering instead."
            )

        # Get the row index for this movie in the similarity matrix
        movie_idx = self.movie_mapper[movie_id]

        # Get similarity scores against all other movies
        sim_scores = list(enumerate(self.item_similarity[movie_idx]))

        # Sort descending, skip the first (self-similarity = 1.0)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        # Map matrix indices back to movieIds
        idx_to_movie_id = {v: k for k, v in self.movie_mapper.items()}

        results = []
        for rank, (idx, score) in enumerate(sim_scores, start=1):
            mid = idx_to_movie_id.get(idx)
            if mid:
                title = self.movie_id_to_title.get(mid, "Unknown")
                results.append({
                    "rank": rank,
                    "title": title,
                    "similarity_score": round(score, 4)
                })

        return pd.DataFrame(results), best_match_title

    # ------------------------------------------------------------------
    # Utility: Predict rating a given user might give a movie
    # ------------------------------------------------------------------
    def predict_rating(self, user_id: int, movie_id: int, top_k: int = 10) -> float:
        """
        Predict what rating user_id would give movie_id.
        
        Uses weighted average of ratings from similar movies the user has rated.
        This is the standard Item-Based CF rating prediction formula.
        
        Returns:
          float: predicted rating (0.5 – 5.0), or None if not predictable
        """
        if user_id not in self.user_item_matrix.index:
            return None
        if movie_id not in self.movie_mapper:
            return None

        movie_idx = self.movie_mapper[movie_id]
        sim_scores = self.item_similarity[movie_idx]

        # Get movies this user has actually rated
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movie_ids = user_ratings[user_ratings > 0].index.tolist()

        # Weighted average: sum(sim * rating) / sum(|sim|)
        numerator = 0.0
        denominator = 0.0
        for rated_mid in rated_movie_ids[:top_k]:
            if rated_mid in self.movie_mapper:
                rated_idx = self.movie_mapper[rated_mid]
                sim = sim_scores[rated_idx]
                rating = user_ratings[rated_mid]
                numerator += sim * rating
                denominator += abs(sim)

        if denominator == 0:
            return None
        return round(numerator / denominator, 2)
