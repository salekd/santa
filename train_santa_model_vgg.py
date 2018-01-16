from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD


# Dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 300
nb_validation_samples = 161
epochs = 5
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Load pretrained VGG16 model.
# The last (top) layers doing the final classification are not included.
vgg16 = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)

# Freeze the weights for the first layers.
for layer in vgg16.layers[:17]:
    layer.trainable = False
vgg16.summary()

# Add custom layers.
x = vgg16.output
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

# Create final model.
model = Model(input = vgg16.input, output = predictions)

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.0001, momentum=0.9),
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

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('santa_vgg.h5')
