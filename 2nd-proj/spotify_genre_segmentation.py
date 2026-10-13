# ============================================================
# Project 2 : Spotify Songs' Genre Segmentation (K-Means)
# Run:  python spotify_genre_segmentation.py   (spotify_dataset.csv in same folder)
# Colab: upload spotify_dataset.csv, paste this code, run.
# Plots are shown AND saved in the "plots" folder.
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png", dpi=110)
    plt.show()

# ------------------------------------------------------------
# STEP 1 : LOAD DATA
# ------------------------------------------------------------
raw = pd.read_csv("spotify_dataset.csv")
print("Shape:", raw.shape)
print(raw.head())
print(raw.info())
print("Missing values:\n", raw.isnull().sum()[raw.isnull().sum() > 0])

# ------------------------------------------------------------
# STEP 2 : DATA PRE-PROCESSING
# ------------------------------------------------------------
df = raw.dropna(subset=["track_name", "track_artist", "track_album_name"]).copy()
print("After dropping null rows:", len(df))

df["release_date"] = pd.to_datetime(df["track_album_release_date"], errors="coerce", format="mixed")
df["release_year"] = df["release_date"].dt.year

# same song can be in many playlists -> keep one row per song for clustering
songs = df.drop_duplicates("track_id").reset_index(drop=True)
print("Duplicate track_id rows:", df.duplicated("track_id").sum(), "| unique songs:", len(songs))

audio = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]

# log transform the skewed columns, then scale
X = songs[audio].copy()
for col in ["speechiness", "instrumentalness", "liveness"]:
    X[col] = np.log1p(X[col])
X_scaled = StandardScaler().fit_transform(X)

# ------------------------------------------------------------
# STEP 3 : DATA ANALYSIS & VISUALIZATION
# ------------------------------------------------------------
print(songs[audio + ["track_popularity"]].describe().T)

# 3.1 Histograms
songs[audio].hist(bins=40, figsize=(14, 8), color="#1db954")
plt.suptitle("Feature distributions")
save("01_histograms")

# 3.2 Genre / sub-genre counts
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.countplot(data=df, y="playlist_genre", order=df["playlist_genre"].value_counts().index, ax=ax[0], color="#1db954")
ax[0].set_title("Tracks per genre")
sns.countplot(data=df, y="playlist_subgenre", order=df["playlist_subgenre"].value_counts().index, ax=ax[1], color="#1db954")
ax[1].set_title("Tracks per sub-genre")
save("02_genre_counts")

# 3.3 Features by genre
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, col in zip(ax.ravel(), ["danceability", "energy", "valence", "acousticness", "speechiness", "tempo"]):
    sns.boxplot(data=songs, x="playlist_genre", y=col, ax=a, palette="viridis")
    a.set_title(col); a.set_xlabel("")
save("03_features_by_genre")

# 3.4 Popularity + release year
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(data=songs, x="playlist_genre", y="track_popularity", ax=ax[0], palette="viridis"); ax[0].set_title("Popularity by genre")
recent = songs.dropna(subset=["release_year"]); recent = recent[recent["release_year"] >= 1980]
recent.groupby("release_year").size().plot(ax=ax[1], color="#1db954"); ax[1].set_title("Songs by release year")
save("04_popularity_year")

# 3.5 Genre profile heatmap
genre_profile = songs.groupby("playlist_genre")[audio].mean()
genre_norm = (genre_profile - genre_profile.min()) / (genre_profile.max() - genre_profile.min())
plt.figure(figsize=(10, 4))
sns.heatmap(genre_norm, annot=genre_profile.round(2), fmt="", cmap="YlGn", cbar=False)
plt.title("Average audio profile of each genre")
save("05_genre_profile")

# ------------------------------------------------------------
# STEP 4 : CORRELATION MATRIX
# ------------------------------------------------------------
corr = songs[audio + ["key", "mode", "track_popularity"]].corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=np.triu(np.ones_like(corr, dtype=bool)))
plt.title("Correlation matrix")
save("06_correlation_matrix")

# ------------------------------------------------------------
# STEP 5 : K-MEANS CLUSTERING
# ------------------------------------------------------------
# 5.1 Elbow + silhouette to choose k
inertia, sil = [], []
k_range = range(2, 13)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_, sample_size=6000, random_state=42))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(list(k_range), inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(k_range), sil, "o-", color="#e8590c"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
save("07_elbow_silhouette")

# 5.2 Final K-Means (k = 6 -> same as the 6 playlist genres, easy to compare)
K = 6
kmeans = KMeans(n_clusters=K, n_init=20, random_state=42).fit(X_scaled)
songs["cluster"] = kmeans.labels_

# 5.3 PCA to see clusters in 2D
pca = PCA(n_components=2, random_state=42)
pts = pca.fit_transform(X_scaled)
songs["pc1"], songs["pc2"] = pts[:, 0], pts[:, 1]
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(data=songs, x="pc1", y="pc2", hue="cluster", palette="tab10", s=6, alpha=0.6, ax=ax[0]); ax[0].set_title("K-Means clusters")
sns.scatterplot(data=songs, x="pc1", y="pc2", hue="playlist_genre", palette="tab10", s=6, alpha=0.6, ax=ax[1]); ax[1].set_title("Actual playlist genre")
for a in ax: a.legend(markerscale=3)
save("08_pca_clusters")

# 5.4 Cluster characteristics
cluster_profile = songs.groupby("cluster")[audio].mean()
cluster_profile["songs"] = songs["cluster"].value_counts().sort_index()
print("\nCluster profile:\n", cluster_profile.round(2))
cp_norm = (cluster_profile[audio] - cluster_profile[audio].min()) / (cluster_profile[audio].max() - cluster_profile[audio].min())
plt.figure(figsize=(10, 4))
sns.heatmap(cp_norm, annot=cluster_profile[audio].round(2), fmt="", cmap="YlGn", cbar=False)
plt.title("Cluster characteristics")
save("09_cluster_profile")

# 5.5 Clusters vs playlist_genre
plt.figure(figsize=(8, 4))
sns.heatmap(pd.crosstab(songs["cluster"], songs["playlist_genre"], normalize="index").round(2), annot=True, cmap="Blues", cbar=False)
plt.title("Share of each genre inside each cluster")
save("10_cluster_vs_genre")
print("Adjusted Rand Index (cluster vs genre):", round(adjusted_rand_score(songs["playlist_genre"], songs["cluster"]), 3))

# 5.6 Clusters vs playlist_subgenre
plt.figure(figsize=(11, 6))
sns.heatmap(pd.crosstab(songs["playlist_subgenre"], songs["cluster"], normalize="index").round(2), annot=True, cmap="Blues", cbar=False)
plt.title("Cluster share inside each sub-genre")
save("11_cluster_vs_subgenre")

# 5.7 Cluster the PLAYLISTS (by average audio profile)
playlists = df.groupby("playlist_name").agg(n=("track_id", "count"), genre=("playlist_genre", "first"),
                                            **{c: (c, "mean") for c in audio})
playlists = playlists[playlists["n"] >= 10]
P_scaled = StandardScaler().fit_transform(playlists[audio])
playlists["cluster"] = KMeans(n_clusters=K, n_init=20, random_state=42).fit_predict(P_scaled)
pp = PCA(n_components=2, random_state=42).fit_transform(P_scaled)
playlists["pc1"], playlists["pc2"] = pp[:, 0], pp[:, 1]
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(data=playlists, x="pc1", y="pc2", hue="cluster", palette="tab10", s=40, ax=ax[0]); ax[0].set_title(f"{len(playlists)} playlists clustered")
sns.scatterplot(data=playlists, x="pc1", y="pc2", hue="genre", palette="tab10", s=40, ax=ax[1]); ax[1].set_title("Same playlists coloured by genre")
save("12_playlist_clusters")
print("Playlist-level ARI vs genre:", round(adjusted_rand_score(playlists["genre"], playlists["cluster"]), 3))

# ------------------------------------------------------------
# STEP 6 : SONG RECOMMENDATION (same cluster + cosine similarity)
# ------------------------------------------------------------
def recommend(track_name, n=10):
    match = songs[songs["track_name"].str.lower() == track_name.lower()]
    if match.empty:
        return "Song not found"
    i = match.index[0]
    same_cluster = songs[songs["cluster"] == songs.loc[i, "cluster"]].index.drop(i)
    sim = cosine_similarity(X_scaled[[i]], X_scaled[same_cluster])[0]
    top = same_cluster[np.argsort(-sim)[:n]]
    out = songs.loc[top, ["track_name", "track_artist", "playlist_genre", "track_popularity"]].copy()
    out["similarity"] = np.sort(sim)[::-1][:n].round(3)
    return out.reset_index(drop=True)

top_song = songs.sort_values("track_popularity", ascending=False).iloc[0]
print(f"\nBecause you like: {top_song['track_name']} - {top_song['track_artist']}")
print(recommend(top_song["track_name"], 8))

songs.to_csv("spotify_clustered.csv", index=False)
print("\nSaved clustered songs -> spotify_clustered.csv")
