import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

class ImageRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
      
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _process_image(self, image_path):
       
        img = imread(image_path)
        
       
        if len(img.shape) == 3:
           
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        
        
        img = self._resize_image(img, (32, 32))
        
        
        features = []
        
       
        features.extend(img.flatten())
        
       
        features.extend([
            np.mean(img),      
            np.std(img),       
            np.max(img),       
            np.min(img),       
            np.median(img),    
        ])
        
        return np.array(features)
    
    def _resize_image(self, image, size):
       
        h, w = image.shape
        h_target, w_target = size
        
       
        x = np.linspace(0, w-1, w_target)
        y = np.linspace(0, h-1, h_target)
        
       
        x_coords, y_coords = np.meshgrid(x, y)
        
       
        x_coords = np.round(x_coords).astype(int)
        y_coords = np.round(y_coords).astype(int)
        
        
        resized = image[y_coords, x_coords]
        
        return resized
    
    def prepare_dataset(self, data_folder):
       
        features = []
        labels = []
        
        
        self.class_features = {}
        
        for class_name in os.listdir(data_folder):
            class_path = os.path.join(data_folder, class_name)
            if not os.path.isdir(class_path):
                continue
            
            print(f"Processing class: {class_name}")
            
            for image_name in os.listdir(class_path):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                image_path = os.path.join(class_path, image_name)
                try:
                    image_features = self._process_image(image_path)
                    features.append(image_features)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
        
        return np.array(features), np.array(labels)
    
    def train(self, data_folder, test_size=0.2):
       
        print("Preparing dataset...")
        X, y = self.prepare_dataset(data_folder)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print("Training Random Forest classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': range(X.shape[1]),
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'accuracy': self.classifier.score(X_test_scaled, y_test),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance.head(10)  # Top 10 important features
        }
        
        return results
    
    def predict(self, image_path):
        
        if not self.is_trained:
            raise Exception("Classifier needs to be trained first")
        
        # Process image
        features = self._process_image(image_path)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probabilities
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Create probability dictionary
        prob_dict = dict(zip(self.classifier.classes_, probabilities))
        
        return prediction, prob_dict
    
    def visualize_results(self, image_path):
       
        prediction, probabilities = self.predict(image_path)
        
        # Create figure with subplots
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        img = imread(image_path)
        plt.imshow(img)
        plt.title('Input Image')
        plt.axis('off')
        
        # Plot prediction probabilities
        plt.subplot(1, 3, 2)
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        plt.bar(classes, probs)
        plt.title('Classification Probabilities')
        plt.xticks(rotation=45)
        plt.ylabel('Probability')
        
        # Plot feature importance
        plt.subplot(1, 3, 3)
        feature_importance = pd.DataFrame({
            'feature': range(len(self.classifier.feature_importances_)),
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        plt.bar(range(10), feature_importance['importance'])
        plt.title('Top 10 Important Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize classifier
    classifier = ImageRandomForestClassifier(n_estimators=100, max_depth=10)
    
    # Example paths - replace with your actual paths
    training_folder = "path/to/training/data"
    test_image = "path/to/test/image.jpg"
    
    # Train classifier
    results = classifier.train(training_folder)
    
    # Print results
    print("\nTraining Results:")
    print(f"Accuracy: {results['accuracy']:.2f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nTop 10 Important Features:")
    print(results['feature_importance'])
    
    # Classify and visualize test image
    prediction, probabilities = classifier.predict(test_image)
    print(f"\nPredicted class: {prediction}")
    print("\nClass probabilities:")
    for class_name, prob in probabilities.items():
        print(f"{class_name}: {prob:.2f}")
    
    # Visualize results
    classifier.visualize_results(test_image)

if __name__ == "__main__":
    main()
