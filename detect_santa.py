from PIL import Image
from keras.models import load_model
import numpy as np
import sys

filename = sys.argv[1]
image = Image.open(filename).convert('RGB')
im_width, im_height = 150, 150
image = image.resize((im_width, im_height), Image.ANTIALIAS)
image_np = np.array(image.getdata()).reshape(
    (1, im_height, im_width, 3)).astype(np.float32) / 255.

model = load_model('santa.h5')
pred = model.predict(image_np)
print(pred)
