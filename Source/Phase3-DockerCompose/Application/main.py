from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from Image_preprocessing import image_preprocessing
from starlette.responses import HTMLResponse
from UNET_tf import unet
from Classifier import classifier, preprocessing_and_predict
from PIL import Image
from io import BytesIO
import numpy as np
import imghdr
import base64
import io
import mlflow
import psutil
import time


app = FastAPI()
classifier_model = classifier()
seg_model = unet()
seg_model.load_weights('unet_membrane.hdf5')
    
mlflow.set_tracking_uri("http://bridgenet:5000")
# Start an MLflow run
mlflow.start_run()
run_name = "Model-A"
mlflow.set_tag("mlflow.runName", run_name)

# Add CORS middleware to allow requests from the web page's origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with the appropriate origin of your web page
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Mount the "static" directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_html():
    with open("static/index.html", "r") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)

# Load and serve the new version of the model
@app.post('/model')
async def model(image: UploadFile = File(...)):
    
    image_bytes = await image.read()
    
    start_time = time.time()

    image_test, check, image_class  = reading_image_properly(image_bytes)

    if check:
        img = preprocess_predict_image(image_test)
        
        # convert numpy array to base64-encoded data URL
        predicted_img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        predicted_img = predicted_img.astype(np.uint8)

        buffer = io.BytesIO()
        predicted_img_pil = Image.fromarray(predicted_img)
        predicted_img_pil.save(buffer, format="PNG")
        predicted_img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        img = predicted_img_str

        cpu_percent , memory_percent = monitor_hardware()
        latency = measure_latency(start_time)
        # Log metrics to MLflow
        mlflow.log_metric("CPU Usage", cpu_percent)
        mlflow.log_metric("Memory Usage", memory_percent)
        mlflow.log_metric("Latency", latency)
        return {"image": str(img), "message" : f"Image Class: {image_class}"}
    else:
        cpu_percent , memory_percent = monitor_hardware()
        latency = measure_latency(start_time)
        # Log metrics to MLflow
        mlflow.log_metric("CPU Usage", cpu_percent)
        mlflow.log_metric("Memory Usage", memory_percent)
        mlflow.log_metric("Latency", latency) 
        return {"image": "Your Uploaded Image was not in the right format",
                "message" : f"Image Class: {image_class}"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    return await model(image)
    
def reading_image_properly(image_bytes):
    
    image_format = imghdr.what(None, image_bytes)
    
    Check = True

    image_class, image_class_name = preprocessing_and_predict(classifier_model, image_bytes)
    
    if image_format is None:
        Check = False
        img = None
    else:
        Check = True    
        pil_image = Image.open(BytesIO(image_bytes)).convert("L")
        img = np.array(pil_image)

    return img, Check, image_class_name
    
def preprocess_predict_image(test_image):
    
    ip = image_preprocessing()
    testGene = ip.testGenerator(test_image)
    
    
    output = seg_model.predict(testGene, 1, verbose=1)
    
    img = ip.saveResult(output)
    
    return img

# Define a function to monitor hardware usage
def monitor_hardware():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    return cpu_percent , memory_percent

# Define a function to measure latency
def measure_latency(start_time):
    latency = time.time() - start_time
    return latency

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)