"""Train the cattle/buffalo classifier.
Usage: python backend/train.py <dataset_dir> [epochs]
dataset_dir needs one folder per class, e.g. Cattle/ and Buffalo/."""
import json, sys
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

data = Path(sys.argv[1])
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)

names = sorted(d.name for d in data.iterdir() if d.is_dir())
paths, labels = [], []
for i, n in enumerate(names):
    for p in sorted((data / n).iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            paths.append(str(p)); labels.append(i)
paths, labels = np.array(paths), np.array(labels)
p_tr, p_rest, y_tr, y_rest = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)
p_val, p_te, y_val, y_te = train_test_split(p_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)
print({"train": len(p_tr), "val": len(p_val), "test": len(p_te)})

aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomZoom(0.15)])

def load(path, y):
    img = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    return img / 127.5 - 1.0, y

def make(p, y, train=False):
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if train:
        ds = ds.shuffle(len(p), seed=42)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    if train:
        ds = ds.map(lambda x, t: (aug(x, training=True), t))
    return ds.prefetch(tf.data.AUTOTUNE)

train, val, test = make(p_tr, y_tr, True), make(p_val, y_val), make(p_te, y_te)
cw = {i: len(y_tr) / (len(names) * int(np.sum(y_tr == i))) for i in range(len(names))}

base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False
x = layers.Dropout(0.3)(layers.GlobalAveragePooling2D()(base.output))
model = models.Model(base.input, layers.Dense(len(names), activation="softmax")(x))
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=epochs, class_weight=cw)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=max(3, epochs // 2), class_weight=cw)

y_pred = np.argmax(model.predict(test, verbose=0), 1)
rep = classification_report(y_te, y_pred, target_names=names, output_dict=True)
metrics = {"accuracy": rep["accuracy"], "per_class": {k: rep[k] for k in names},
           "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
           "test_images": int(len(y_te)), "backbone": "MobileNetV2 (ImageNet transfer learning)"}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
(out / "classes.json").write_text(json.dumps(names))
model.save(out / "gokuleye.keras")
print(f"Test accuracy: {rep['accuracy']:.3f}  on {len(y_te)} test images")
print(confusion_matrix(y_te, y_pred))
