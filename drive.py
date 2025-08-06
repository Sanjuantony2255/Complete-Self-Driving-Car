import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
from keras.losses import MeanSquaredError  # Explicitly import MeanSquaredError
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize SocketIO server
sio = socketio.Server()

# Initialize Flask app
app = Flask(__name__)  # '__main__'

# Speed limit for throttle calculation
speed_limit = 10

# Image preprocessing function
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian Blur
    img = cv2.resize(img, (200, 66))  # Resize to model input size
    img = img / 255  # Normalize pixel values
    return img

# Event listener for telemetry
@sio.on('telemetry')
def telemetry(sid, data):
    print(f"Received data: {data}")  # Log the received data

    if 'image' in data:
        # Extract speed and image data
        speed = float(data['speed'])
        print(f"Speed: {speed}")  # Debug speed

        # Decode and preprocess the image
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])

        # Predict steering angle
        steering_angle = float(model.predict(image))
        throttle = 1.0 - speed / speed_limit  # Calculate throttle

        print(f"Steering angle: {steering_angle}, Throttle: {throttle}")
        send_control(steering_angle, throttle)
    else:
        print("No image data received")

# Event listener for client connection
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# Function to send control commands
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    # Load the model and handle custom objects
    model = load_model('model/model.h5', custom_objects={'mse': MeanSquaredError()})

    # Wrap Flask app with SocketIO middleware
    app = socketio.Middleware(sio, app)

    # Start the server
    print("Server is starting...")
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
