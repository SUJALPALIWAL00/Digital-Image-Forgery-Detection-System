import os
import numpy as np
from sklearn.model_selection import train_test_split
from ela import convert_to_ela
from model import build_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def load_data(data_path):
    X, Y = [], []
    for label_name, label_value in [('Real', 0), ('Fake', 1)]:
        folder_path = os.path.join(data_path, label_name)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Loading {len(files)} images from {label_name}...")
        for img_name in files:
            try:
                img_path = os.path.join(folder_path, img_name)
                ela_img = convert_to_ela(img_path).resize((128, 128))
                X.append(img_to_array(ela_img) / 255.0)
                Y.append(label_value)
            except: continue
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, Y = load_data('data')
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

    # Strong Augmentation for 2000 images
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Callbacks: Automatically stop if model stops improving
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    model = build_model()
    
    print("🚀 Training Final Model...")
    model.fit(
        datagen.flow(X_train, Y_train, batch_size=32),
        validation_data=(X_val, Y_val), 
        epochs=100, 
        callbacks=[early_stop, reduce_lr]
    )

    model.save('trained_model.h5')
    print("✅ Optimized model saved as 'trained_model.h5'.")