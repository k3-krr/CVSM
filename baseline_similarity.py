import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_FILE = r"D:\DataSet\cvsm_embeddings.npz"

data = np.load(EMBEDDING_FILE)

embeddings = data["embeddings"]
song_ids = data["song_ids"]

print("Embedding matrix:", embeddings.shape)

similarity = cosine_similarity(embeddings)

print("\nCVSM COSINE SIMILARITY")
print("=" * 50)

for i in range(len(song_ids)):
    for j in range(i + 1, len(song_ids)):
        print(
            f"{song_ids[i]} <-> {song_ids[j]} : "
            f"{similarity[i][j]:.4f}"
        )