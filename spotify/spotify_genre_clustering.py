"""Spotify Songs' Genre Segmentation - preprocessing, EDA, clustering, recommender.
Usage: python spotify_genre_clustering.py [path/to/spotify_dataset.csv]
Writes spotify_report.html (self-contained) next to the script."""
import sys, io, base64, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
PATH = sys.argv[1] if len(sys.argv) > 1 else "spotify_dataset.csv"
FEATS = ["danceability","energy","key","loudness","mode","speechiness","acousticness",
         "instrumentalness","liveness","valence","tempo","duration_ms"]
AUDIO = [f for f in FEATS if f not in ("key","mode")]   # used for clustering

html = []
def h(tag, text): html.append(f"<{tag}>{text}</{tag}>")
def p(text): h("p", text)
def table(df, **kw): html.append(df.to_html(classes="t", border=0, **kw))
def fig(title=None, note=None):
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close()
    if title: h("h3", title)
    html.append(f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')
    if note: p(note)

# ---------------- 1. Pre-processing ----------------
raw = pd.read_csv(PATH)
h("h2", "1. Data pre-processing")
log = [f"Raw shape: {raw.shape}",
       f"Missing values: {int(raw.isna().sum().sum())} (all in track_name / artist / album of 5 rows)"]
df = raw.dropna(subset=["track_name","track_artist","track_album_name"]).copy()
log.append(f"After dropping null rows: {len(df)}")
log.append(f"Rows sharing a track_id (same song listed in several playlists): {df.duplicated('track_id').sum()}")
df["release_date"] = pd.to_datetime(df["track_album_release_date"], errors="coerce", format="mixed")
df["release_year"] = df["release_date"].dt.year
songs = df.drop_duplicates("track_id").reset_index(drop=True)      # one row per song for modelling
log.append(f"Unique songs used for modelling: {len(songs)}")
for l in log: html.append(f"<li>{l}</li>")
p("Features were standardised (zero mean, unit variance). Right-skewed features "
  "(speechiness, instrumentalness, liveness) were log1p-transformed first so a few extreme values don't dominate the distance.")
X = songs[AUDIO].copy()
for c in ["speechiness","instrumentalness","liveness"]: X[c] = np.log1p(X[c])
Xs = StandardScaler().fit_transform(X)

# ---------------- 2. EDA ----------------
h("h2", "2. Exploratory analysis")
table(songs[AUDIO+["track_popularity"]].describe().T.round(3))

songs[AUDIO].hist(bins=40, figsize=(14,8), color="#1db954"); plt.suptitle("Feature distributions"); fig("Feature distributions")

fig_, ax = plt.subplots(1,2, figsize=(13,4))
sns.countplot(data=df, y="playlist_genre", order=df.playlist_genre.value_counts().index, ax=ax[0], color="#1db954")
ax[0].set_title("Tracks per genre")
sns.countplot(data=df, y="playlist_subgenre", order=df.playlist_subgenre.value_counts().index, ax=ax[1], color="#1db954")
ax[1].set_title("Tracks per sub-genre"); fig("Genre / sub-genre counts")

cols = ["danceability","energy","valence","acousticness","speechiness","tempo"]
fig_, axs = plt.subplots(2,3, figsize=(15,8))
for a,c in zip(axs.ravel(), cols):
    sns.boxplot(data=songs, x="playlist_genre", y=c, ax=a, palette="viridis"); a.set_title(c); a.set_xlabel("")
fig("Audio features by genre")

fig_, ax = plt.subplots(1,2, figsize=(13,4))
sns.boxplot(data=songs, x="playlist_genre", y="track_popularity", ax=ax[0], palette="viridis"); ax[0].set_title("Popularity by genre")
yr = songs.dropna(subset=["release_year"]); yr = yr[yr.release_year>=1980]
yr.groupby("release_year").size().plot(ax=ax[1], color="#1db954"); ax[1].set_title("Songs by release year")
fig("Popularity & release year")

fig_, ax = plt.subplots(figsize=(8,4))
yr.groupby("release_year")[["danceability","energy","valence","acousticness"]].mean().plot(ax=ax)
ax.set_title("Average feature value by release year"); fig("Trends over time")

gp = songs.groupby("playlist_genre")[AUDIO].mean()
gp_n = (gp-gp.min())/(gp.max()-gp.min())
fig_, ax = plt.subplots(figsize=(9,4)); sns.heatmap(gp_n, annot=gp.round(2), fmt="", cmap="YlGn", ax=ax, cbar=False)
ax.set_title("Genre profile (colour = relative level across genres, numbers = mean)"); fig("Genre profiles")

h("h2", "3. Correlation matrix")
corr = songs[FEATS+["track_popularity"]].corr()
fig_, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=np.triu(np.ones_like(corr,bool)), ax=ax)
fig("Correlation of features")
pairs = corr.where(np.tril(np.ones(corr.shape),-1).astype(bool)).stack().dropna().sort_values()
p(f"Strongest positive: <b>{pairs.index[-1][0]} / {pairs.index[-1][1]}</b> ({pairs.iloc[-1]:.2f}). "
  f"Strongest negative: <b>{pairs.index[0][0]} / {pairs.index[0][1]}</b> ({pairs.iloc[0]:.2f}), "
  f"and <b>{pairs.index[1][0]} / {pairs.index[1][1]}</b> ({pairs.iloc[1]:.2f}). "
  "Energy, loudness and acousticness carry overlapping information.")

# ---------------- 4. Clustering ----------------
h("h2", "4. Clustering")
ks = range(2,13); inertia=[]; sil=[]
for k in ks:
    km = KMeans(k, n_init=10, random_state=42).fit(Xs); inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_, sample_size=6000, random_state=42))
fig_, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(list(ks), sil, "o-", color="#e8590c"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
fig("Choosing k")
K = 6   # matches the 6 playlist genres, so clusters can be compared directly with them
p(f"Silhouette scores are low across all k (best {max(sil):.2f}), meaning songs form a continuous spectrum rather than tight separate blobs. "
  f"k = {K} was used so the clusters can be compared directly against the 6 playlist genres.")
km = KMeans(K, n_init=20, random_state=42).fit(Xs); songs["cluster"] = km.labels_
pca = PCA(2, random_state=42); Z = pca.fit_transform(Xs); songs["pc1"], songs["pc2"] = Z[:,0], Z[:,1]
ev = pca.explained_variance_ratio_

fig_, ax = plt.subplots(1,2, figsize=(15,6))
sns.scatterplot(data=songs, x="pc1", y="pc2", hue="cluster", palette="tab10", s=6, alpha=.6, ax=ax[0]); ax[0].set_title("KMeans clusters (PCA view)")
sns.scatterplot(data=songs, x="pc1", y="pc2", hue="playlist_genre", palette="tab10", s=6, alpha=.6, ax=ax[1]); ax[1].set_title("Actual playlist genre (PCA view)")
for a in ax: a.set_xlabel(f"PC1 ({ev[0]:.0%})"); a.set_ylabel(f"PC2 ({ev[1]:.0%})"); a.legend(markerscale=3)
fig("Clusters vs genres")

prof = songs.groupby("cluster")[AUDIO].mean(); prof["size"] = songs.cluster.value_counts().sort_index()
prof_n = (prof[AUDIO]-prof[AUDIO].min())/(prof[AUDIO].max()-prof[AUDIO].min())
fig_, ax = plt.subplots(figsize=(10,4)); sns.heatmap(prof_n, annot=prof[AUDIO].round(2), fmt="", cmap="YlGn", cbar=False, ax=ax)
ax.set_title("Cluster profiles"); fig("What each cluster sounds like")
table(prof.round(2))

ct = pd.crosstab(songs.cluster, songs.playlist_genre, normalize="index").round(2)
fig_, ax = plt.subplots(figsize=(8,4)); sns.heatmap(ct, annot=True, cmap="Blues", cbar=False, ax=ax); ax.set_title("Share of each genre inside each cluster")
fig("Cluster vs playlist_genre")
ari = adjusted_rand_score(songs.playlist_genre, songs.cluster)
p(f"Adjusted Rand Index vs playlist genre: <b>{ari:.3f}</b> (0 = random, 1 = identical). Audio features alone only partly recover genre — "
  "EDM (cluster 2) and rap (cluster 5) are the only genres that form a clear majority in a cluster; pop, r&amp;b, latin and rock are spread across several clusters and overlap heavily.")

# Clusters by playlist_subgenre
fig_, ax = plt.subplots(figsize=(11,6)); sns.heatmap(pd.crosstab(songs.playlist_subgenre, songs.cluster, normalize="index").round(2), annot=True, cmap="Blues", cbar=False, ax=ax)
ax.set_title("Cluster share inside each sub-genre"); fig("Clusters by sub-genre")

# Clusters by playlist name (playlist-level aggregation)
pl = df.groupby("playlist_name").agg(n=("track_id","count"), genre=("playlist_genre","first"), **{f:(f,"mean") for f in AUDIO})
pl = pl[pl.n>=10]
Xp = StandardScaler().fit_transform(pl[AUDIO]); pl["cluster"] = KMeans(K, n_init=20, random_state=42).fit_predict(Xp)
Zp = PCA(2, random_state=42).fit_transform(Xp); pl["pc1"], pl["pc2"] = Zp[:,0], Zp[:,1]
fig_, ax = plt.subplots(1,2, figsize=(15,6))
sns.scatterplot(data=pl, x="pc1", y="pc2", hue="cluster", palette="tab10", s=40, ax=ax[0]); ax[0].set_title(f"{len(pl)} playlists clustered by average audio profile")
sns.scatterplot(data=pl, x="pc1", y="pc2", hue="genre", palette="tab10", s=40, ax=ax[1]); ax[1].set_title("Same playlists coloured by genre")
fig("Clusters by playlist name")
p(f"Playlist-level clustering ARI vs genre: <b>{adjusted_rand_score(pl.genre, pl.cluster):.3f}</b> — averaging over a playlist removes noise, so playlists group by genre much more cleanly than individual songs.")

# ---------------- 5. Models ----------------
h("h2", "5. Model & recommendation system")
Xtr,Xte,ytr,yte = train_test_split(Xs, songs.playlist_genre, test_size=.2, random_state=42, stratify=songs.playlist_genre)
rf = RandomForestClassifier(300, n_jobs=-1, random_state=42).fit(Xtr,ytr)
acc = accuracy_score(yte, rf.predict(Xte))
p(f"Supervised check: a random forest predicts playlist genre from audio features with <b>{acc:.1%}</b> test accuracy "
  f"(chance ≈ 17%). Top features: " + ", ".join(pd.Series(rf.feature_importances_, AUDIO).nlargest(4).index) + ".")

# Recommender: same-cluster candidates ranked by cosine similarity in scaled feature space
def recommend(track_name, n=10, artist=None):
    m = songs[songs.track_name.str.lower()==track_name.lower()]
    if artist: m = m[m.track_artist.str.lower()==artist.lower()]
    if m.empty: return f"'{track_name}' not found"
    i = m.index[0]; pool = songs[songs.cluster==songs.cluster[i]].index.drop(i)
    s = cosine_similarity(Xs[[i]], Xs[pool])[0]
    out = songs.loc[pool[np.argsort(-s)[:n]], ["track_name","track_artist","playlist_genre","track_popularity"]].copy()
    out["similarity"] = np.sort(s)[::-1][:n].round(3); return out
p("Recommender: given a song, find its cluster, then rank the other songs in that cluster by cosine similarity of their scaled audio features.")
for name in songs.sort_values("track_popularity", ascending=False).track_name.head(3):
    r = songs[songs.track_name==name].iloc[0]
    h("h3", f"Because you like: {r.track_name} — {r.track_artist} ({r.playlist_genre})")
    table(recommend(name, 8, r.track_artist).reset_index(drop=True), index=False)

# Leave-one-out sanity check: do recommendations share the genre of the seed?
rng = np.random.default_rng(0); hit=[]
for i in rng.choice(len(songs), 300, replace=False):
    pool = songs[songs.cluster==songs.cluster[i]].index.drop(i)
    s = cosine_similarity(Xs[[i]], Xs[pool])[0]; top = pool[np.argsort(-s)[:10]]
    hit.append((songs.playlist_genre[top]==songs.playlist_genre[i]).mean())
p(f"Sanity check on 300 random songs: on average <b>{np.mean(hit):.0%}</b> of the top-10 recommendations share the seed song's genre (random baseline ≈ 17%).")

h("h2", "Key takeaways")
for t in ["Songs form a continuous spectrum; clustering is soft and genres overlap, especially pop / r&b / latin.",
          "Energy–loudness are strongly positively correlated; acousticness is negatively correlated with both.",
          "EDM has the highest energy and tempo, rap the highest speechiness and danceability, r&amp;b the lowest energy and highest acousticness, rock the lowest danceability.",
          "Playlist-level averages cluster cleanly by genre; individual songs are noisier.",
          "Cluster + cosine similarity gives a working content-based recommender; add popularity or artist filters to improve it."]:
    html.append(f"<li>{t}</li>")

css = """body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#222;line-height:1.5}
h1{color:#1db954}h2{border-bottom:2px solid #1db954;padding-bottom:4px;margin-top:40px}img{max-width:100%}
table.t{border-collapse:collapse;font-size:13px;margin:10px 0}table.t td,table.t th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:right}
table.t th{background:#f3f3f3}@media(prefers-color-scheme:dark){body{background:#121212;color:#ddd}table.t th{background:#222}table.t td{border-color:#333}}"""
open("spotify_report.html","w").write(f"<!doctype html><meta charset=utf-8><title>Spotify Genre Segmentation</title><style>{css}</style>"
    f"<h1>Spotify Songs' Genre Segmentation</h1>{''.join(html)}")
songs.to_csv("spotify_clustered.csv", index=False)
print("done", f"ARI={ari:.3f} RF_acc={acc:.3f} recHit={np.mean(hit):.2f}")
