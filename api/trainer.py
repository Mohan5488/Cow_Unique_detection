import os
import cv2
import numpy as np
import hashlib
# ORB and Haar Cascade init
orb = cv2.ORB_create()
descriptors = []
labels = []
label_dict = {}
label_idx = 0

nose_cascade = cv2.CascadeClassifier('api/cascade.xml')

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

def extract_nose(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noses = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(noses) == 0:
        return None
    x, y, w, h = noses[0]
    return img[y:y+h, x:x+w]

def convert_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted_image = cv2.bitwise_not(adaptive_thresh)
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)
    processed_image = cv2.resize(processed_image, (256, 256))
    return processed_image

def get_unique_id(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()[:8]

def train_on_image(image_path, name, location):
    global label_idx
    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not read image"

    nose_img = extract_nose(img)
    if nose_img is None:
        return None, "No nose detected"

    processed_img = convert_image(nose_img)
    keypoints, des = orb.detectAndCompute(processed_img, None)
    if des is not None:
        image_bytes = cv2.imencode('.jpg', processed_img)[1].tobytes()
        unique_id = get_unique_id(image_bytes)

        descriptors.append(des)
        label_dict[label_idx] = {
            'unique_id' : unique_id,
            'name' : name,
            'location' : location
        }
        labels.extend([label_idx] * len(des))
        label_idx += 1

        return unique_id,label_dict, "Training successful"
    else:
        return None, {}, "No features detected"
    
def finalize_training():
    global descriptors, labels, label_dict

    if not descriptors:
        return "No descriptors collected"

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Paths
    model_path = os.path.join(output_dir, "cowrec_knn_model.xml")
    label_dict_path = os.path.join(output_dir, "label_dict.npy")
    descriptors_path = os.path.join(output_dir, "descriptors.npy")
    labels_path = os.path.join(output_dir, "labels.npy")

    # Convert new training data
    new_descriptors = np.vstack(descriptors).astype(np.float32)
    new_labels = np.array(labels)

    # Initialize combined arrays
    combined_descriptors = new_descriptors
    combined_labels = new_labels

    # Load previously saved data if available
    if os.path.exists(descriptors_path) and os.path.exists(labels_path):
        old_descriptors = np.load(descriptors_path)
        old_labels = np.load(labels_path)

        # Combine old and new
        combined_descriptors = np.vstack((old_descriptors, new_descriptors))
        combined_labels = np.hstack((old_labels, new_labels))

    # Retrain KNN from combined data
    knn = cv2.ml.KNearest_create()
    knn.train(combined_descriptors, cv2.ml.ROW_SAMPLE, combined_labels)

    # Save model and all data
    knn.save(model_path)
    np.save(descriptors_path, combined_descriptors)
    np.save(labels_path, combined_labels)
    np.save(label_dict_path, label_dict)

    # Clear current session’s in-memory training data
    descriptors.clear()
    labels.clear()

    return "KNN model, descriptors, labels, and label dictionary updated and saved"
