from django.shortcuts import get_object_or_404, render, redirect
from django.core.files.storage import FileSystemStorage
import os
import cv2
import numpy as np
import hashlib
from .models import Hombenai
# Create your views here.

orb = cv2.ORB_create()
descriptors = []
labels = []
label_dict = {}
label_idx = 0

model_path = "output/cowrec_knn_model.xml"
label_dict_path = "output/label_dict.npy"
cascade_path = "api/cascade.xml"

if not (os.path.exists(model_path) and os.path.exists(label_dict_path)):
    raise FileNotFoundError("Model or label dictionary not found. Please train first.")

knn = cv2.ml.KNearest_create()
knn = knn.load(model_path)
label_dict = np.load(label_dict_path, allow_pickle=True).item()

orb = cv2.ORB_create()
nose_cascade = cv2.CascadeClassifier(cascade_path)


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
    label_dict_path = os.path.join('output', 'label_dict.npy')
    if os.path.exists(label_dict_path):
        saved_dict = np.load(label_dict_path, allow_pickle=True).item()
        label_dict.update(saved_dict)
        if label_dict:
            label_idx = max(label_dict.keys()) + 1
    
    img = cv2.imread(image_path)
    if img is None:
        return None,{}, "Could not read image"

    nose_img = extract_nose(img)
    if nose_img is None:
        return None,{}, "No nose detected"

    processed_img = convert_image(nose_img)
    keypoints, des = orb.detectAndCompute(processed_img, None)
    print(des)
    
    if des is not None:
        image_bytes = cv2.imencode('.jpg', processed_img)[1].tobytes()
        unique_id = get_unique_id(image_bytes)

        for i in label_dict:
            if label_dict[i]['unique_id'] == unique_id:
                return None,{},"Already exists"
            
        descriptors.append(des)
        label_dict[label_idx] = {
            'unique_id' : unique_id,
            'name' : name,
            'location' : location
        }
        labels.extend([label_idx] * len(des))
        
        
        relative_image_path = os.path.relpath(image_path, start='media')
        Hombenai.objects.create(
            unique_id=unique_id,
            name=name,
            location=location,
            upload=relative_image_path
        )
        
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

    model_path = os.path.join(output_dir, "cowrec_knn_model.xml")
    label_dict_path = os.path.join(output_dir, "label_dict.npy")
    descriptors_path = os.path.join(output_dir, "descriptors.npy")
    labels_path = os.path.join(output_dir, "labels.npy")

    new_descriptors = np.vstack(descriptors).astype(np.float32)
    new_labels = np.array(labels)

    combined_descriptors = new_descriptors
    combined_labels = new_labels
    if os.path.exists(descriptors_path) and os.path.exists(labels_path):
        old_descriptors = np.load(descriptors_path)
        old_labels = np.load(labels_path)

        combined_descriptors = np.vstack((old_descriptors, new_descriptors))
        combined_labels = np.hstack((old_labels, new_labels))
        
    knn = cv2.ml.KNearest_create()
    knn.train(combined_descriptors, cv2.ml.ROW_SAMPLE, combined_labels)

    knn.save(model_path)
    np.save(descriptors_path, combined_descriptors)
    np.save(labels_path, combined_labels)
    np.save(label_dict_path, label_dict)

    descriptors.clear()
    labels.clear()
    
    return "KNN model, descriptors, labels, and label dictionary updated and saved"


def predict_id(image_path):
    global label_dict
    knn = cv2.ml.KNearest_create()
    knn = knn.load("output/cowrec_knn_model.xml")

    img = cv2.imread(image_path)
    if img is None:
        return None, "Invalid image"

    nose_img = extract_nose(img)
    if nose_img is None:
        return None, "Nose not detected"

    processed_img = convert_image(nose_img)
    keypoints, des = orb.detectAndCompute(processed_img, None)
    if des is None:
        return None, "No features found"

    des = des.astype(np.float32)
    ret, results, neighbours, dist = knn.findNearest(des, k=1)

    votes = results.flatten().astype(int)
    best_label = np.bincount(votes).argmax()
    unique_id = label_dict.get(best_label, "Unknown")

    return unique_id, "Prediction successful"

def home(request):
    return render(request, 'index.html')

def training(request):
    objs = Hombenai.objects.all()
    return render(request, 'Train_cow.html',context={'objs':objs})
def testing(request):
    return render(request, 'test.html')

def train(request):
    message = finalize_training()
    objs = Hombenai.objects.all()
    return render(request, 'Train_cow.html',{
                    'message': "Training done successfully",
                    'objs' : objs
                })

def upload(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        location = request.POST.get('location')
        uploaded_file = request.FILES.get('image')

        if uploaded_file:
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            unique_id,label_dict, message = train_on_image(file_path, name, location)
            
            objs = Hombenai.objects.all()
            if unique_id:
                return render(request, 'Train_cow.html', {
                    'message': f"Training successful. Unique ID: {unique_id}",
                    'name': name,
                    'location': location,
                    'file_url': fs.url(filename),
                    'objs' : objs
                })
            else:
                return render(request, 'Train_cow.html', {
                    'error': message,
                    'name': name,
                    'location': location,
                    'objs' : objs
                }) 
        else:
            print('Failed to upload')

    return render(request, 'index.html')


def test_image(request):
    if request.method == 'POST' and request.FILES.get('testfile'):
        uploaded_file = request.FILES['testfile']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        predicted_id, message = predict_id(file_path)
        name = predicted_id['name']
        unique_id = predicted_id['unique_id']
        location = predicted_id['location']
        
        
        return render(request, 'test.html', {
            'file_url': fs.url(filename),
            'name' :name,
            'unique_id':unique_id,
            'location':location,
            'message': message
        })

    return render(request, 'test.html')


def retrain_after_deletion():
    global label_dict
    descriptors_path = 'output/descriptors.npy'
    labels_path = 'output/labels.npy'
    model_path = 'output/cowrec_knn_model.xml'

    if not (os.path.exists(descriptors_path) and os.path.exists(labels_path)):
        return "Descriptors or labels not found"

    all_descriptors = np.load(descriptors_path)
    all_labels = np.load(labels_path)

    # Only keep descriptors whose label is still in label_dict
    filtered_des = []
    filtered_labels = []

    for des, label in zip(all_descriptors, all_labels):
        if label in label_dict:
            filtered_des.append(des)
            filtered_labels.append(label)

    if not filtered_des:
        return "No descriptors left to retrain"

    filtered_des = np.array(filtered_des, dtype=np.float32)
    filtered_labels = np.array(filtered_labels, dtype=np.int32)

    # Save filtered data
    np.save(descriptors_path, filtered_des)
    np.save(labels_path, filtered_labels)

    # Retrain KNN
    knn = cv2.ml.KNearest_create()
    knn.train(filtered_des, cv2.ml.ROW_SAMPLE, filtered_labels)
    knn.save(model_path)

    return "Model retrained successfully after deletion"


def delete_cow(request, id):
    global label_dict
    if request.method == 'POST':
        obj = get_object_or_404(Hombenai, id=id)
        unique_id = obj.unique_id
        obj.delete()

        # Remove from label_dict
        keys_to_delete = [k for k, v in label_dict.items() if v['unique_id'] == unique_id]
        for k in keys_to_delete:
            del label_dict[k]

        # Save updated label_dict
        np.save("output/label_dict.npy", label_dict)

        # Retrain the model
        message = retrain_after_deletion()

    return redirect('training')
