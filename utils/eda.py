# =============================================================================
# utils/eda.py
# Purpose: Exploratory Data Analysis — understand the dataset before modeling
# EDA is a CRITICAL step in any ML project. It reveals data distributions,
# missing values, biases, and helps decide what features matter most.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Use non-interactive backend for server/script environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# Professional plot style
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#333355",
    "axes.labelcolor": "#c8c8e8",
    "xtick.color": "#c8c8e8",
    "ytick.color": "#c8c8e8",
    "text.color": "#c8c8e8",
    "grid.color": "#2a2a4a",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})

ACCENT = "#7c6af7"
ACCENT2 = "#f76c6c"
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


def print_summary(movies, ratings):
    """Print a clean text summary of the dataset."""
    print("\n" + "="*60)
    print("  📊  DATASET SUMMARY")
    print("="*60)
    print(f"\n🎬 Movies DataFrame Shape  : {movies.shape}")
    print(f"⭐ Ratings DataFrame Shape : {ratings.shape}")
    print(f"\n--- Movies Sample ---")
    print(movies[["movieId", "clean_title", "genres", "year"]].head(5).to_string(index=False))
    print(f"\n--- Ratings Distribution ---")
    print(ratings["rating"].describe().round(2))
    print(f"\n--- Missing Values (Movies) ---")
    print(movies.isnull().sum())
    print(f"\n--- Missing Values (Ratings) ---")
    print(ratings.isnull().sum())
    print("="*60 + "\n")


def plot_eda(movies, ratings):
    """
    Generate a comprehensive EDA dashboard and save it as a PNG.
    
    Plots:
      1. Rating value distribution (histogram)
      2. Ratings per user (how active are users?)
      3. Ratings per movie (popularity distribution — long tail!)
      4. Top 20 most-rated movies
      5. Top genres by movie count
      6. Average rating by genre
    """
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    # --- Plot 1: Rating Distribution ---
    ax1 = fig.add_subplot(gs[0, :2])
    rating_counts = ratings["rating"].value_counts().sort_index()
    bars = ax1.bar(rating_counts.index, rating_counts.values,
                   color=ACCENT, edgecolor="#0f0f1a", linewidth=0.5, width=0.35)
    ax1.set_title("⭐ Rating Distribution", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Rating Value")
    ax1.set_ylabel("Count")
    ax1.set_xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    # Annotate bars
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 200, f"{int(h):,}",
                 ha="center", va="bottom", fontsize=7, color="#c8c8e8")

    # --- Plot 2: Ratings per User (Log scale to handle skew) ---
    ax2 = fig.add_subplot(gs[0, 2])
    user_counts = ratings.groupby("userId").size()
    ax2.hist(user_counts, bins=40, color=ACCENT2, edgecolor="#0f0f1a", linewidth=0.5)
    ax2.set_title("👥 Ratings per User", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("# Ratings")
    ax2.set_ylabel("# Users")
    ax2.set_yscale("log")

    # --- Plot 3: Ratings per Movie (long-tail distribution) ---
    ax3 = fig.add_subplot(gs[1, :2])
    movie_counts = ratings.groupby("movieId").size()
    ax3.hist(movie_counts, bins=50, color="#4ecdc4", edgecolor="#0f0f1a", linewidth=0.5)
    ax3.set_title("🎬 Ratings per Movie (Long Tail)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("# Ratings Received")
    ax3.set_ylabel("# Movies")
    ax3.set_yscale("log")
    # Add annotation explaining long tail
    ax3.text(0.65, 0.85, "Most movies\nare rarely rated\n(long tail effect)",
             transform=ax3.transAxes, fontsize=9, color="#ffcc44",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2a4a", alpha=0.8))

    # --- Plot 4: Top 20 Most-Rated Movies ---
    ax4 = fig.add_subplot(gs[1, 2])
    top_movies = (ratings.groupby("movieId").size()
                  .reset_index(name="count")
                  .merge(movies[["movieId", "clean_title"]], on="movieId")
                  .nlargest(10, "count"))
    short_titles = top_movies["clean_title"].str[:20] + "…"
    ax4.barh(range(len(top_movies)), top_movies["count"].values,
             color=ACCENT, edgecolor="#0f0f1a")
    ax4.set_yticks(range(len(top_movies)))
    ax4.set_yticklabels(short_titles, fontsize=8)
    ax4.invert_yaxis()
    ax4.set_title("🏆 Top 10 Most Rated", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("# Ratings")

    # --- Plot 5: Genre Frequency ---
    ax5 = fig.add_subplot(gs[2, :2])
    # Explode the genres list column into individual rows
    genre_series = movies["genres_list"].explode()
    genre_counts = genre_series[genre_series != ""].value_counts().head(15)
    bars5 = ax5.bar(genre_counts.index, genre_counts.values,
                    color=ACCENT2, edgecolor="#0f0f1a", linewidth=0.5)
    ax5.set_title("🎭 Movie Count by Genre", fontsize=13, fontweight="bold", pad=10)
    ax5.set_xlabel("Genre")
    ax5.set_ylabel("# Movies")
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # --- Plot 6: Avg Rating by Genre ---
    ax6 = fig.add_subplot(gs[2, 2])
    merged_genres = ratings.merge(movies[["movieId", "genres_list"]], on="movieId")
    merged_genres = merged_genres.explode("genres_list")
    genre_avg = (merged_genres.groupby("genres_list")["rating"]
                 .mean().sort_values(ascending=True).tail(12))
    ax6.barh(genre_avg.index, genre_avg.values, color="#4ecdc4", edgecolor="#0f0f1a")
    ax6.set_title("⭐ Avg Rating by Genre", fontsize=13, fontweight="bold", pad=10)
    ax6.set_xlabel("Avg Rating")
    ax6.set_xlim(3.0, 4.2)
    for i, v in enumerate(genre_avg.values):
        ax6.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    # Master title
    fig.suptitle("🎬  MovieLens Dataset — Exploratory Data Analysis",
                 fontsize=18, fontweight="bold", y=1.01, color="#ffffff")

    save_path = os.path.join(STATIC_DIR, "eda_dashboard.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📈 EDA dashboard saved → {save_path}")
    return save_path
