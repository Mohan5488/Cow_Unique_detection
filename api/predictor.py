# import os
# import cv2
# import numpy as np
# import hashlib

# # ==== Load Model & Label Dict ====
# model_path = "output/cowrec_knn_model.xml"
# label_dict_path = "output/label_dict.npy"
# cascade_path = "api/cascade.xml"

# if not (os.path.exists(model_path) and os.path.exists(label_dict_path)):
#     raise FileNotFoundError("Model or label dictionary not found. Please train first.")

# # Load KNN model and label dictionary
# knn = cv2.ml.KNearest_create()
# knn = knn.load(model_path)
# label_dict = np.load(label_dict_path, allow_pickle=True).item()

# print(label_dict)

# # ORB and Haar
# orb = cv2.ORB_create()
# nose_cascade = cv2.CascadeClassifier(cascade_path)

# def extract_nose(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     noses = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
#     if len(noses) == 0:
#         return None
#     x, y, w, h = noses[0]
#     return img[y:y+h, x:x+w]

# def convert_image(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     adaptive_thresh = cv2.adaptiveThreshold(
#         gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     inverted_image = cv2.bitwise_not(adaptive_thresh)
#     kernel = np.ones((3, 3), np.uint8)
#     processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)
#     processed_image = cv2.resize(processed_image, (256, 256))
#     return processed_image

# def predict_id(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         return None, "Invalid image"

#     nose_img = extract_nose(img)
#     if nose_img is None:
#         return None, "Nose not detected"

#     processed_img = convert_image(nose_img)
#     keypoints, des = orb.detectAndCompute(processed_img, None)
#     if des is None:
#         return None, "No features found"

#     des = des.astype(np.float32)
#     ret, results, neighbours, dist = knn.findNearest(des, k=1)

#     votes = results.flatten().astype(int)
#     best_label = np.bincount(votes).argmax()
#     unique_id = label_dict.get(best_label, "Unknown")

#     return unique_id, "Prediction successful"

