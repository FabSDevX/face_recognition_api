from PIL import Image
from flask import Flask, request, jsonify
from joblib import load
import cv2 as cv
import face_recognition as fc
import logging
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
loaded_knn_clf = load("security_face_classification.joblib")

def predict(image, model):
    """
    Realiza predicción del nombre de la persona a partir de una imagen.
    """
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    face_locations = fc.face_locations(rgb_image)
    face_encodings = fc.face_encodings(rgb_image, face_locations)

    names = []

    for face_encoding in face_encodings:
        matches = model.kneighbors([face_encoding], n_neighbors=1)
        name = model.predict([face_encoding])[0] if matches[0][0][0] < 0.5 else "Unknown"
        names.append(name)

    # Devuelve el primer nombre encontrado o "Unknown"
    return names[0] if names else "Unknown"


@app.route('/', methods=['GET'])
def hello():
    return jsonify({"message": "Hello, welcome to the face recognition API!"})
    

@app.route('/login', methods=['POST'])
def login():
    """
    Endpoint para procesar el login basado en reconocimiento facial.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']

    try:
        # Cargar y convertir la imagen a un formato compatible con OpenCV
        image = Image.open(image_file.stream)
        image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        logging.info("Image loaded and converted successfully.")
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({"error": "Error processing image"}), 500

    try:
        # Predecir el nombre utilizando el modelo KNN
        person_name = predict(image, loaded_knn_clf)
        logging.info(f"Prediction result: {person_name}")

        if person_name == "Unknown":
            return jsonify({"found": False, "message": "Persona no registrada."}), 200
        else:
            return jsonify({"found": True, "message": "Persona encontrada", "name": person_name}), 200
    except Exception as e:
        logging.error(f"Error predicting face: {str(e)}")
        return jsonify({"error": f"Error predicting face: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

