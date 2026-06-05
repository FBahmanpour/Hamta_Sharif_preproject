import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import layers, models



DATASET_PATH = "Dataset"
VIDEO_DIR = os.path.join(DATASET_PATH, "videos")
FRAME_ROOT = "frames"
LABEL_PATH = os.path.join(DATASET_PATH, "labels.csv")

IMG_SIZE = (224, 224)
TIMESTEPS = 16
STRIDE = 8


df = pd.read_csv(LABEL_PATH, sep=';')
df["Name"] = df["Name"].str.strip()

label_map = dict(zip(df["Name"], df["Speed"]))


def extract_frames(video_path, output_folder, fps_sample=10):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(video_fps / fps_sample), 1)

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            cv2.imwrite(
                os.path.join(output_folder, f"frame_{saved:04d}.jpg"),
                frame
            )
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"{os.path.basename(video_path)} → {saved} frames")


video_dir = os.path.join(DATASET_PATH, "videos")

for video_file in os.listdir(video_dir):
    if not video_file.endswith(".mp4"):
        continue

    video_name = os.path.splitext(video_file)[0]

    video_path = os.path.join(video_dir, video_file)
    output_folder = os.path.join(FRAME_ROOT, video_name)

    extract_frames(video_path, output_folder)



def load_frames(folder):
    frames = []
    files = sorted(os.listdir(folder))

    for f in files:
        path = os.path.join(folder, f)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        frames.append(img)

    return np.array(frames)



def create_clips(frames, clip_len=TIMESTEPS, stride=STRIDE):
    clips = []

    for i in range(0, len(frames) - clip_len + 1, stride):
        clips.append(frames[i:i + clip_len])

    return np.array(clips)



def build_dataset(video_names):
    X = []
    y = []

    for name in video_names:
        folder = os.path.join(FRAME_ROOT, name)

        if name not in label_map:
            continue

        frames = load_frames(folder)
        if len(frames) < TIMESTEPS:
            continue

        clips = create_clips(frames)

        for clip in clips:
            X.append(clip)
            y.append(label_map[name])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)



video_names = [
    v for v in os.listdir(FRAME_ROOT)
    if os.path.isdir(os.path.join(FRAME_ROOT, v))
    and v in label_map
    ]

train_videos, temp_videos = train_test_split(video_names, test_size=0.3, random_state=42)
val_videos, test_videos = train_test_split(temp_videos, test_size=0.5, random_state=42)


X_train, y_train = build_dataset(train_videos)
X_val, y_val = build_dataset(val_videos)
X_test, y_test = build_dataset(test_videos)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

cnn = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg',
    weights='imagenet'
)

cnn.trainable = False

model = models.Sequential([
    layers.TimeDistributed(cnn, input_shape=(TIMESTEPS, 224, 224, 3)),
    layers.LSTM(64),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='mse',
    metrics=['mae']
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

start_train = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=4,
    callbacks=callbacks,
    shuffle=True
)

train_time = time.time() - start_train

print("\n===== TRAINING TIME =====")
print(f"Training time: {train_time:.2f} seconds")


test_loss, test_mae = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n===== TEST RESULTS =====")
print(f"Test MSE : {test_loss:.6f}")
print(f"Test MAE : {test_mae:.6f}")


predictions = model.predict(X_test)
results_df = pd.DataFrame({
    "Actual_Speed": y_test,
    "Predicted_Speed": predictions.flatten()
})

results_df.to_csv(
    "speed_predictions.csv",
    index=False
)

print("\n===== SPEED PREDICTION =====")
print(results_df.head())

