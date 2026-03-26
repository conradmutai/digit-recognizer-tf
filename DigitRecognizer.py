import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
import sklearn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ==========================================
# Data Preparation
# ==========================================
print("Loading data...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(f"The shape of the training set is: {train_data.shape}")
print(f"The shape of the test set is: {test_data.shape}")

Y_train = train_data["label"]
X_train = train_data.drop(labels=["label"], axis=1)

# Normalization
print("Normalizing data...")
X_train = X_train / 255.0
test_data = test_data / 255.0

# Reshape
print("Reshaping arrays...")
X_train = X_train.values.reshape(-1, 28, 28, 1)
test_data = test_data.values.reshape(-1, 28, 28, 1)

# Label Encoding
Y_train = to_categorical(Y_train, num_classes=10)

# Splitting Into Training and Validation Set
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

# ==========================================
# Convolutional Neural Network
# ==========================================
print("Building the model...")
model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', padding='Same', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (5,5), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (5,5), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

# Optimizers and Annealers
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 30
batch_size = 64

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.5, min_lr=0.00001)

# Data Augmentation
print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1, 
    height_shift_range=0.1
)

datagen.fit(X_train)

# Fitting the Model
print("Training the model...")
history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, Y_val),
    callbacks=[reduce_lr]
)

# ==========================================
# Data Analysis & Visualizations
# ==========================================
print("Plotting training history...")
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['accuracy'], label='training accuracy', color='b')
ax[0].plot(history.history['val_accuracy'], label='validation accuracy', color='r')
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['loss'], label='training loss', color='b')
ax[1].plot(history.history['val_loss'], label='validation loss', color='r')
legend = ax[1].legend(loc='best', shadow=True)
plt.show()

# Predictions
print("Generating validation predictions...")
Y_pred_probs = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred_probs, axis=1)
Y_true = np.argmax(Y_val, axis=1)

# Confusion Matrix
cm = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Error Analysis
print("Plotting prediction errors...")
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred_probs[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

plt.figure(figsize=(12, 8))
for i in range(10):
    if i < len(X_val_errors): # Prevent out-of-bounds if fewer than 10 errors occur
        plt.subplot(2, 5, i+1)
        plt.imshow(X_val_errors[i].reshape(28,28), cmap='gray')
        plt.title(f"True: {Y_true_errors[i]}\nPred: {Y_pred_classes_errors[i]}")
        plt.axis('off')
plt.tight_layout()
plt.show()

# ==========================================
# Submission
# ==========================================
print("Running inference on test dataset...")
results = model.predict(test_data)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("cnn_mnist_submission.csv", index=False)
print("Submission file 'cnn_mnist_submission.csv' has been created!")