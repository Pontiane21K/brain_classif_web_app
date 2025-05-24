from flask import Flask, request, render_template
import os
import torch
from torchvision import transforms
from PIL import Image
import tensorflow as tf

# Configuration Flask
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Classes
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ----------- Chargement des modèles ------------

# PyTorch
from cnn_model import CNN_PyTorch
from processing import preprocess_image
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_PyTorch()
model.load_state_dict(torch.load("/home/students/Documents/Computer vision/image_classification_app/models/Pontiane_model.torch", map_location=device))
model.to(device)
model.eval()

# TensorFlow
model_tf = tf.keras.models.load_model("/home/students/Documents/Computer vision/image_classification_app/models/Pontian_model.tensorflow")

# -----------  Fonctions de prédiction ------------

def predict_pytorch(model, image_path, device, class_names):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]


def predict_tensorflow(model, image_path, class_names):
    image = preprocess_image(image_path)  #
    predictions = model.predict(image)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return class_names[predicted_class]



# -----------  Routes Flask ------------
tumor_info = {
    "glioma": {
        "desc": "Gliomas are brain tumors that originate in the glial cells that support and protect the brain's neurons.",
        "advice": "It's important to consult a neurologist as soon as possible. Treatment may include surgery, radiotherapy or chemotherapy, depending on the type and stage of the glioma.."
    },
    "meningioma": {
        "desc": "Meningiomas are generally benign (non-cancerous) tumors that develop in the meninges, the membranes that surround and protect the brain and spinal cord.",
        "advice": "Regular medical supervision is often recommended. In some cases, surgery may be required if the tumor causes symptoms or grows."
    },
    "notumor": {
        "desc": "No tumor detected. The brain appears healthy.",
        "advice": "Maintain a healthy lifestyle and regular check-ups if necessary."
    },
    "pituitary": {
        "desc": "Tumors of the pituitary gland can disrupt hormone production in the body, as this gland regulates many hormonal functions.",
        "advice": "An endocrinologist may recommend hormone treatment or surgery, depending on the size and impact of the tumour."
    }
}

@app.route('/', methods=['GET', 'POST'])
# -----------  Tumor Information Dictionary ------------
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    prediction = None
    image_name = None
    info = None  # For tumor information

    if request.method == 'POST':
        framework = request.form.get('framework')

        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        if file:
            image_name = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
            file.save(filepath)

            if framework == 'pytorch':
                prediction = predict_pytorch(model, filepath, device, class_names)
            elif framework == 'tensorflow':
                prediction = predict_tensorflow(model_tf, filepath, class_names)

            # Get info for the predicted tumor type
            info = tumor_info.get(prediction, {"desc": "Information not available.", "advice": ""})

    return render_template('index.html', prediction=prediction, image_name=image_name, info=info)



    #return render_template('index.html', prediction=prediction)
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ----------- Lancement de l'application ------------

if __name__ == '__main__':
    app.run(debug=True)
