import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def build_model(input_shape=(224, 224, 3), base_model_trainable=False):
    """Enhance the CNN model with a pre-trained MobileNetV2 and custom layers."""
    # Load MobileNetV2 as the base model
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = base_model_trainable
    
    # Create a new model on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='linear')  # Output layer for embeddings
    ])
    
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Load an image, detect the face, and resize it to the target size."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()

    detections = detector.detect_faces(image)
    if detections:
        x, y, width, height = detections[0]['box']
        cropped_face = image[y:y+height, x:x+width]
        cropped_face = cv2.resize(cropped_face, target_size)
        cropped_face = img_to_array(cropped_face)
        cropped_face = np.expand_dims(cropped_face, axis=0)
        cropped_face = preprocess_input(cropped_face)
        return cropped_face
    else:
        return None

def generate_embeddings(face_image, model):
    """Generate embeddings for a face image using the model."""
    embeddings = model.predict(face_image)
    return embeddings

def load_my_model(model_path=None):
    """Load the CNN model. If no path is provided, build a new model."""
    if model_path:
        model = load_model(model_path)
    else:
        model = build_model()  # Build and return a new model with enhanced features
    return model

# Example usage
if __name__ == "__main__":
    model = load_my_model()  # Load or build the CNN model

    face_image = preprocess_image('path/to/face/image.jpg')
    if face_image is not None:
        embeddings = generate_embeddings(face_image, model)
        print("Generated Embeddings:", embeddings)
    else:
        print("No face detected.")
