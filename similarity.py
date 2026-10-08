import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
data = np.load(r"D:\DataSet\cvsm_embeddings.npz")

song_ids = data["song_ids"]
embeddings = data["embeddings"]

# Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings)

# --------------------------------------------------
# 1. Full similarity matrix with song IDs
# --------------------------------------------------

matrix_df = pd.DataFrame(
    similarity_matrix,
    index=song_ids,
    columns=song_ids
)

print("\n========================================")
print("CVSM SIMILARITY MATRIX")
print("========================================")

print(matrix_df.round(3).to_string())

# Save readable matrix
matrix_df.to_csv(
    r"D:\DataSet\cvsm_similarity_matrix.csv"
)

# --------------------------------------------------
# 2. Find most similar different song for each song
# --------------------------------------------------

results = []

for i in range(len(song_ids)):

    # Copy similarities for this song
    scores = similarity_matrix[i].copy()

    # Ignore itself
    scores[i] = -1

    # Find highest similarity
    best_index = np.argmax(scores)

    results.append({
        "Song": song_ids[i],
        "Most Similar Song": song_ids[best_index],
        "Similarity": scores[best_index]
    })

nearest_df = pd.DataFrame(results)

print("\n========================================")
print("MOST SIMILAR SONG FOR EACH SONG")
print("========================================")

print(nearest_df.to_string(index=False))

nearest_df.to_csv(
    r"D:\DataSet\cvsm_most_similar.csv",
    index=False
)

# --------------------------------------------------
# 3. Create all unique song pairs
# --------------------------------------------------

pairs = []

for i in range(len(song_ids)):
    for j in range(i + 1, len(song_ids)):

        pairs.append({
            "Song A": song_ids[i],
            "Song B": song_ids[j],
            "Similarity": similarity_matrix[i][j]
        })

pairs_df = pd.DataFrame(pairs)

# Sort highest similarity first
pairs_df = pairs_df.sort_values(
    "Similarity",
    ascending=False
)

print("\n========================================")
print("TOP 20 MOST SIMILAR PAIRS")
print("========================================")

print(pairs_df.head(20).to_string(index=False))

pairs_df.to_csv(
    r"D:\DataSet\cvsm_all_pairs.csv",
    index=False
)

print("\n========================================")
print("FILES SAVED")
print("========================================")
print(r"D:\DataSet\cvsm_similarity_matrix.csv")
print(r"D:\DataSet\cvsm_most_similar.csv")
print(r"D:\DataSet\cvsm_all_pairs.csv")