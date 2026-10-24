"""Train the cattle/buffalo classifier.
Usage: python backend/train.py <dataset_dir> [epochs]
dataset_dir needs one folder per class, e.g. Cattle/ and Buffalo/."""
import json, sys
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

data = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)

kw = dict(image_size=(224, 224), batch_size=32, seed=42, validation_split=0.3)
train = tf.keras.utils.image_dataset_from_directory(data, subset="training", **kw)
rest = tf.keras.utils.image_dataset_from_directory(data, subset="validation", **kw)
names = train.class_names
n = rest.cardinality().numpy()
val, test = rest.take(n // 2), rest.skip(n // 2)  # validation and held-out test halves

norm = lambda x, y: (x / 127.5 - 1.0, y)
aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomZoom(0.15)])
train = train.map(lambda x, y: (aug(x, training=True), y)).map(norm).prefetch(2)
val, test = val.map(norm), test.map(norm)

base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False
x = layers.Dropout(0.3)(layers.GlobalAveragePooling2D()(base.output))
model = models.Model(base.input, layers.Dense(len(names), activation="softmax")(x))
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=epochs)

base.trainable = True  # fine-tune the last blocks
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=max(3, epochs // 2))

y_true = np.concatenate([y.numpy() for _, y in test])
y_pred = np.argmax(model.predict(test), 1)
rep = classification_report(y_true, y_pred, target_names=names, output_dict=True)
metrics = {"accuracy": rep["accuracy"], "per_class": {k: rep[k] for k in names},
           "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
           "test_images": int(len(y_true)), "backbone": "MobileNetV2 (ImageNet transfer learning)"}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
(out / "classes.json").write_text(json.dumps(names))
model.save(out / "gokuleye.keras")
print(f"Test accuracy: {rep['accuracy']:.3f}")
