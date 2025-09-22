from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import io
from azure.storage.blob import BlobServiceClient
app = Flask(__name__)

# Azure Blob Storage configuration
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=aicatdogstoraceacct;AccountKey=DUrDegscg7K9SVY/pXLg9ocgBfDx0AH/xCxsxOUFuoH4mZeVVc0kzxDX7+MuFtXJ2qcTALyqIV6c+ASto+hrow==;BlobEndpoint=https://aicatdogstoraceacct.blob.core.windows.net/;FileEndpoint=https://aicatdogstoraceacct.file.core.windows.net/;QueueEndpoint=https://aicatdogstoraceacct.queue.core.windows.net/;TableEndpoint=https://aicatdogstoraceacct.table.core.windows.net"
AZURE_CONTAINER_NAME = "aicatdogcontainer"
AZURE_BLOB_NAME = "animal_classifier_model.h5"
LOCAL_MODEL_PATH = "animal_classifier_model.h5"

# Download model from Azure Blob Storage
def download_model_from_blob():
   blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
   blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=AZURE_BLOB_NAME)
   with open(LOCAL_MODEL_PATH, "wb") as model_file:
       blob_data = blob_client.download_blob()
       blob_data.readinto(model_file)
# Ensure model is downloaded and loaded
download_model_from_blob()
model = load_model(LOCAL_MODEL_PATH)

# Update this list based on your training classes
class_labels = ['cat', 'dog','other']  # Add more if needed

def predict_image(image_path):
   img = load_img(image_path, target_size=(150, 150))
   img_array = img_to_array(img) / 255.0
   img_array = np.expand_dims(img_array, axis=0)
   prediction = model.predict(img_array)
   predicted_class = class_labels[np.argmax(prediction)]
   return predicted_class

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       image = request.files['image']
       if image:
           image_path = os.path.join('static', image.filename)
           image.save(image_path)
           result = predict_image(image_path)
           return render_template('index.html', prediction=result, image_path=image_path)
   return render_template('index.html', prediction=None)
if __name__ == '__main__':
    app.run(debug=True)