from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.vgg16 import VGG16


# Dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 300
nb_validation_samples = 161
epochs = 2
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Load pretrained VGG16 model.
# The last (top) layers doing the final classification are not included.
vgg16 = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)
vgg16.summary()

# Add custom layers.
model = Sequential()
model.add(Flatten(input_shape=vgg16.layers[-1].output_shape[1:]))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30.,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for (x_train, y_train), (x_validation, y_validation) in zip(train_generator, validation_generator):
        pred_train = vgg16.predict(x_train)
        pred_validation = vgg16.predict(x_validation)
        model.fit(pred_train, y_train, validation_data=(pred_validation, y_validation))

        batches += 1
        # We need to break the loop by hand because the generator loops indefinitely.
        if batches > nb_train_samples // batch_size:
            break

model.save('santa_vgg_bottleneck.h5')
