from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# Constants - update paths & classes as needed
TRAIN_DIR = "../fashion_data/train"
VALIDATION_DIR = "../fashion_data/validation"
NUM_CLASSES = 5  # Adjust this to your actual classes count
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (224, 224)

def train():
    # Data Augmentation for training (keeps it real, no overfitting)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = build_model(NUM_CLASSES)

    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Save the trained model
    model.save("trained_model.h5")
    print("Model training completed and saved as trained_model.h5")

if __name__ == "__main__":
    train()
