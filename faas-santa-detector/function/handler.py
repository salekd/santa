import json
from json import encoder
from PIL import Image
from io import BytesIO
from keras.models import load_model
import numpy as np


def handle(req):  
    image = Image.open(BytesIO(req)).convert('RGB')
    im_width, im_height = 150, 150
    image = image.resize((im_width, im_height), Image.ANTIALIAS)
    image_np = np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.float32) / 225.

    model = load_model('/root/function/santa.h5')
    pred = model.predict(image_np)

    encoder.FLOAT_REPR = lambda f: format(f, '.4f')
    encoder.c_make_encoder = None
    result = {}
    result['detected_objects'] = [
            {'class': 'santa', 'score': float(pred[0][0])},
            {'class': 'not_santa', 'score': 1. - float(pred[0][0])}]

    print(json.dumps(result))
