import os
import numpy as np
import librosa
import tensorflow as tf

from mscol import network
from cola import constants


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

MODEL_DIR = r"models/htdemucs/m4all_cvsmah"
AUDIO_DIR = r"D:\DataSet\test_audio"
OUTPUT_FILE = r"D:\DataSet\cvsm_embeddings.npz"


# --------------------------------------------------
# 2. Load pretrained CVSM model
# --------------------------------------------------

print("Loading CVSM model...")

contrastive_network = network.get_contrastive_network(
    embedding_dim=512,
    temperature=0.2,
    pooling_type='max',
    similarity_type=constants.SimilarityMeasure.BILINEAR
)

contrastive_network.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
)

checkpoint = tf.train.latest_checkpoint(MODEL_DIR)

print("Checkpoint:", checkpoint)

contrastive_network.load_weights(checkpoint).expect_partial()

encoder = contrastive_network.embedding_model.get_layer("encoder")

print("Encoder loaded:", encoder.name)
print("Encoder output:", encoder.output_shape)


# --------------------------------------------------
# 3. Create CVSM preprocessing model
# --------------------------------------------------

inputs = tf.keras.layers.Input(shape=(16000,))

x = tf.signal.stft(
    inputs,
    frame_length=400,
    frame_step=160,
    fft_length=1024
)

x = tf.abs(x)

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    64,
    x.shape[-1],
    16000,
    60,
    7800
)

x = tf.tensordot(
    x,
    linear_to_mel_weight_matrix,
    1
)

x = tf.clip_by_value(
    x,
    clip_value_min=1e-5,
    clip_value_max=1e8
)

x = tf.expand_dims(
    tf.math.log(x),
    axis=-1
)

print("CVSM input shape:", x.shape)

model = tf.keras.Model(inputs, encoder(x))


# --------------------------------------------------
# 4. Find all MP3 files
# --------------------------------------------------

audio_files = [
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith(".mp3")
]

audio_files.sort()

print("\nFound", len(audio_files), "audio files")

if len(audio_files) == 0:
    print("No MP3 files found.")
    exit()


# --------------------------------------------------
# 5. Generate embeddings
# --------------------------------------------------

embeddings = []
song_ids = []

segment_length = 16000

for count, filename in enumerate(audio_files, start=1):

    audio_path = os.path.join(AUDIO_DIR, filename)

    print("\n----------------------------------------")
    print("Processing", count, "/", len(audio_files))
    print("File:", filename)

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    print("Sample rate:", sr)
    print("Duration:", len(audio) / sr, "seconds")

    # Take first 1 second
    if len(audio) < segment_length:
        audio = np.pad(
            audio,
            (0, segment_length - len(audio))
        )

    segment = audio[:segment_length]

    # Generate embedding
    embedding = model.predict(
        np.expand_dims(segment, axis=0),
        verbose=0
    )[0]

    print("Embedding shape:", embedding.shape)
    print("First 5 values:", embedding[:5])

    embeddings.append(embedding)

    # Remove .mp3 from filename
    song_id = os.path.splitext(filename)[0]
    song_ids.append(song_id)


# --------------------------------------------------
# 6. Convert to NumPy arrays
# --------------------------------------------------

embeddings = np.asarray(embeddings)
song_ids = np.asarray(song_ids)

print("\n========================================")
print("CVSM BASELINE COMPLETE")
print("========================================")

print("Number of songs:", len(song_ids))
print("Embedding matrix shape:", embeddings.shape)
print("Embedding dimension:", embeddings.shape[1])


# --------------------------------------------------
# 7. Save embeddings
# --------------------------------------------------

np.savez(
    OUTPUT_FILE,
    song_ids=song_ids,
    embeddings=embeddings
)

print("\nSaved embeddings to:")
print(OUTPUT_FILE)