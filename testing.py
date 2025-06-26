import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def load_images_as_vectors(folder_path, image_size=(100, 100)):
    image_vectors = []
    image_labels = []
    label_to_name = {}

    for label, person_dir in enumerate(sorted(os.listdir(folder_path))):
        person_path = os.path.join(folder_path, person_dir)
        if not os.path.isdir(person_path):
            continue
        label_to_name[label] = person_dir  # Map label to folder name
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            img_vector = img.flatten()
            image_vectors.append(img_vector)
            image_labels.append(label)

    if len(image_vectors) == 0:
        raise ValueError("No images found! Check your dataset path and folder structure.")

    Face_Db = np.array(image_vectors).T  # shape: (mn, p)
    Labels = np.array(image_labels)      # shape: (p,)
    return Face_Db, Labels, label_to_name


def calculate_mean_face(Face_Db):
    mean_face = np.mean(Face_Db, axis=1).reshape(-1, 1)
    return mean_face


def mean_zero_faces(Face_Db, mean_face):
    Delta = Face_Db - mean_face
    return Delta


def calculate_surrogate_covariance(Delta):
    C = np.dot(Delta.T, Delta)  
    return C


def eigen_decomposition(C):
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors


def get_top_k_eigenvectors(Delta, eigenvectors, k):
    eigenfaces = np.dot(Delta, eigenvectors)
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])
    feature_vectors = eigenfaces[:, :k]  
    return feature_vectors


def generate_eigenfaces(feature_vectors):
    return feature_vectors.T  


def generate_signatures(eigenfaces, Delta):
    signatures = np.dot(eigenfaces, Delta)  
    return signatures


def train_ann(signatures, Labels):
    X = signatures.T  
    y = Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
    return model

# Testing Functions 

def preprocess_and_project_test_image(test_img_path, mean_face, eigenfaces, image_size=(100, 100)):
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(test_img, image_size)
    test_vector = test_img.flatten().reshape(-1, 1)
    test_mean_zero = test_vector - mean_face
    omega = np.dot(eigenfaces, test_mean_zero)  
    return omega.T  

def predict_test_image(test_img_path, mean_face, eigenfaces, model):
    test_signature = preprocess_and_project_test_image(test_img_path, mean_face, eigenfaces)
    predicted_label = model.predict(test_signature)[0]
    return predicted_label

def evaluate_accuracy_with_different_k(Face_Db, Labels, mean_face, k_values):
    accuracies = []
    for k in k_values:
        Delta = mean_zero_faces(Face_Db, mean_face)
        C = calculate_surrogate_covariance(Delta)
        eigenvalues, eigenvectors = eigen_decomposition(C)
        feature_vectors = get_top_k_eigenvectors(Delta, eigenvectors, k)
        eigenfaces = generate_eigenfaces(feature_vectors)
        signatures = generate_signatures(eigenfaces, Delta)

        X = signatures.T
        y = Labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"k = {k} ➜ Accuracy = {acc:.4f}")

    plt.plot(k_values, accuracies, marker='o', linestyle='--', color='blue')
    plt.title("Accuracy vs. k (Number of Eigenfaces)")
    plt.xlabel("Number of Eigenfaces (k)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

# Main Execution 
if __name__ == "__main__":
    folder = "dataset/faces"  

    print("🔄 Loading and processing images...")
    Face_Db, Labels, label_to_name = load_images_as_vectors(folder)

    print("📐 Calculating mean face and centering data...")
    mean_face = calculate_mean_face(Face_Db)
    Delta = mean_zero_faces(Face_Db, mean_face)

    print("📊 Calculating surrogate covariance...")
    C = calculate_surrogate_covariance(Delta)

    print("🔍 Performing eigen decomposition...")
    eigenvalues, eigenvectors = eigen_decomposition(C)

    k = 50  
    print(f"📈 Selecting top {k} eigenvectors...")
    feature_vectors = get_top_k_eigenvectors(Delta, eigenvectors, k)

    print("🧠 Generating eigenfaces...")
    eigenfaces = generate_eigenfaces(feature_vectors)

    print("🧬 Creating face signatures...")
    signatures = generate_signatures(eigenfaces, Delta)

    print("🤖 Training ANN model...")
    model = train_ann(signatures, Labels)


    print("\n🔬 Testing with a single image...")
    test_img_path = "dataset/faces/Akshay/face_12.jpg"  
    predicted = predict_test_image(test_img_path, mean_face, eigenfaces, model)
    print(f"Predicted Class: {predicted}")
    predicted_name = label_to_name.get(predicted, "Unknown")
    print(f"Predicted Class: {predicted} ➜ Person: {predicted_name}")



    k_values = [10, 20, 30, 40, 50, 75, 100]
    evaluate_accuracy_with_different_k(Face_Db, Labels, mean_face, k_values)
