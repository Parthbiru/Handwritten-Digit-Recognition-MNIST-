# mnist_cnn.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import random
import os

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess
# Expand dims (channels), normalize, one-hot labels
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# add channel dimension
x_train = np.expand_dims(x_train, -1)  # shape: (60000,28,28,1)
x_test  = np.expand_dims(x_test, -1)   # shape: (10000,28,28,1)

num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)

# 3. Build a small CNN
def build_model(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Callbacks
es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
mc = callbacks.ModelCheckpoint('mnist_cnn_best.h5', monitor='val_accuracy', save_best_only=True)

# 5. Optional: simple data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.05,
    zoom_range=0.08
)
datagen.fit(x_train)

# 6. Train
batch_size = 128
epochs = 30

history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=batch_size, seed=SEED),
    steps_per_epoch = x_train.shape[0] // batch_size,
    epochs = epochs,
    validation_data = (x_test, y_test_cat),
    callbacks=[es, rlrop, mc],
    verbose=2
)

# 7. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

# 8. Save final model
model.save('mnist_cnn_final.h5')
print("Saved model to mnist_cnn_final.h5")

# 9. Plot training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('epoch'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.xlabel('epoch'); plt.legend()
plt.show()

# 10. Visualize some predictions
def show_predictions(model, x, y_true, num=12):
    idxs = np.random.choice(len(x), num, replace=False)
    preds = model.predict(x[idxs])
    pred_labels = np.argmax(preds, axis=1)
    true_labels = y_true[idxs]

    plt.figure(figsize=(12,4))
    for i, idx in enumerate(idxs):
        plt.subplot(2, 6, i+1)
        plt.imshow(x[idx].squeeze(), cmap='gray')
        plt.axis('off')
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        plt.title(f"P:{pred_labels[i]} / T:{true_labels[i]}", color=color)
    plt.tight_layout()
    plt.show()

show_predictions(model, x_test, y_test, num=12)
