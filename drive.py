import base64
from io import BytesIO

import cv2
import eventlet
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model


MODEL_PATH = "models/experiment_4_model.keras"
MAX_SPEED = 20.0
MIN_SPEED = 8.0


sio = socketio.Server()
app = Flask(__name__)
model = load_model(MODEL_PATH)


def preprocess_image(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0
    return image


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        },
    )


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        sio.emit("manual", data={})
        return

    speed = float(data["speed"])

    image = Image.open(
        BytesIO(base64.b64decode(data["image"]))
    )

    image = np.asarray(image)
    image = preprocess_image(image)

    steering_angle = float(
        model.predict(
            np.array([image]),
            verbose=0,
        )[0][0]
    )

    # Slow down for sharper turns.
    turn_strength = abs(steering_angle)

    if turn_strength < 0.05:
        target_speed = 20.0
    elif turn_strength < 0.12:
        target_speed = 14.0
    else:
        target_speed = 9.0

    throttle = 1.0 - (speed / target_speed)

    throttle = max(-1.0, min(1.0, throttle))

    print(
        f"Steering: {steering_angle:.3f} | "
        f"Speed: {speed:.2f} | "
        f"Target: {target_speed:.1f} | "
        f"Throttle: {throttle:.3f}"
    )

    send_control(
        steering_angle,
        throttle,
    )


@sio.on("connect")
def connect(sid, environ):
    print("Simulator connected")
    send_control(0, 0)


if __name__ == "__main__":
    app = socketio.Middleware(
        sio,
        app,
    )

    eventlet.wsgi.server(
        eventlet.listen(("", 4567)),
        app,
    )