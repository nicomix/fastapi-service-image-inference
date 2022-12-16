import os
from fastapi import APIRouter, File, UploadFile

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import numpy as np

from io import BytesIO

import urllib.request

from pydantic import BaseModel

import cv2

model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

router = APIRouter()

class ImagesUrl(BaseModel):
    urls: list[str]

def loadImage(URL):
    with urllib.request.urlopen(URL) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(125, 125))

    return image.img_to_array(img)

def extractFeatures(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    probs = model(x)
    class_id = tf.argmax(probs, axis=-1)
    return {
        "predicted_class": class_id.numpy().tolist()[0],
        "probs": model.predict(x)[0].tolist()
    }

def extractFeacturesBatch(images):
    x = []
    for url in images.urls:
        image = loadImage(url)
        image = cv2.resize(image,(224,224))
        x.append(image)
    x = np.asarray(x)
    x = preprocess_input(x)
    probs = model(x)
    class_ids = tf.argmax(probs, axis=-1)
    response = []
    for count, class_id in enumerate(class_ids.numpy().tolist()):
        response.append(
            {
                "image_name": "image_" + str(count),
                "predicted_class": class_id,
                "probs": probs.numpy().tolist()[count],
            }
        )
    return response

@router.post("/single-inference")
async def singleInference(file: UploadFile):
    try:
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        file.file.close()
    return extractFeatures(tmp_path)

@router.post("/batch-inference")
async def batchInference(images: ImagesUrl):
    return extractFeacturesBatch(images)
