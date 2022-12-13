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
