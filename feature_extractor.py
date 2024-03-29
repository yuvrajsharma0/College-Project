from keras.preprocessing import image
from keras.applications.resnet import ResNet50, preprocess_input
import numpy as np


class FeatureExtractor:
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')

    @staticmethod
    def extract(img_path):
        # Loading Image using image path
        img = image.load_img(img_path)
        
        # ResNet50 must take a 224x224 img as an input
        img = img.resize((224, 224))
        
        # Make sure img is color
        img = img.convert('RGB')

        # To np.array. Height x Width x Channel. dtype = float32
        img_array = image.img_to_array(img)

        # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        expanded_img_array = np.expand_dims(img_array, axis=0)

        # Subtracting avg values for each pixel
        preprocessed_img = preprocess_input(expanded_img_array)

        # (1, 2048) -> (2048)
        features = FeatureExtractor.model.predict(preprocessed_img)  

        flattened_features = features.flatten()

        # Normalize
        normalized_features = flattened_features / np.linalg.norm(flattened_features)

        # Return data
        return normalized_features
