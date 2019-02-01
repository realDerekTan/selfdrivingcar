# import modules
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, concatenate
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as k
from keras.preprocessing import image
from keras.models import load_model
from keras_tqdm import TQDMNotebookCallback
import os
from Generator import DriveDataGenerator
from Cooking import checkAndCreateDir
import h5py
from PIL import ImageDraw
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# replace these directories with your own
COOKED_DATA_DIR = "C:\\Derek\\TKS\\AirSim\\data_cooked"
MODEL_OUTPUT_DIR = "C:\\Derek\\TKS\\AirSim\\model_output"

train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')
test_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'test.h5'), 'r')

num_train_examples = train_dataset['image'].shape[0]
num_eval_examples = eval_dataset['image'].shape[0]
num_test_examples = test_dataset['image'].shape[0]

batch_size = 32

# use neural nets to generate data by changing values
data_generator = DriveDataGenerator(rescale=1./255., horizontal_flip=True, brighten_range=0.4)
train_generator = data_generator.flow\
    (train_dataset['image'], train_dataset['previous_state'], train_dataset['label'], batch_size=batch_size,
     zero_drop_percentage=0.95, roi=[76, 135, 0, 255])
eval_generator = data_generator.flow\
    (eval_dataset['image'], eval_dataset['previous_state'], eval_dataset['label'], batch_size=batch_size,
     zero_drop_percentage=0.95, roi=[76, 135, 0, 255])


# predict steering angles
def draw_image_with_label(img, label, prediction=None):
    theta = label * 0.69  # Steering range for the car is +- 40 degrees -> 0.69 radians
    line_length = 50
    line_thickness = 3
    label_line_color = (255, 0, 0)
    prediction_line_color = (0, 0, 255)
    pil_image = image.array_to_img(img, k.image_data_format(), scale=True)
    print('Actual Steering Angle = {0}'.format(label))
    draw_image = pil_image.copy()
    image_draw = ImageDraw.Draw(draw_image)
    first_point = (int(img.shape[1]/2), img.shape[0])
    second_point = (int((img.shape[1]/2) + (line_length * math.sin(theta))),
                    int(img.shape[0] - (line_length * math.cos(theta))))
    image_draw.line([first_point, second_point], fill=label_line_color, width=line_thickness)

    if prediction is not None:
        print('Predicted Steering Angle = {0}'.format(prediction))
        print('L1 Error: {0}'.format(abs(prediction-label)))
        theta = prediction * 0.69
        second_point = (int((img.shape[1]/2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length *
                                                                                                     math.cos(theta))))
        image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)

    """
    del image_draw
    plt.imshow(draw_image)
    plt.show()
    """


[sample_batch_train_data, sample_batch_test_data] = next(train_generator)
for i in range(0, 3, 1):
    draw_image_with_label(sample_batch_train_data[0][i], sample_batch_test_data[i])

image_input_shape = sample_batch_train_data[0].shape[1:]
state_input_shape = sample_batch_train_data[1].shape[1:]
activation = 'relu'

# Create the convolutional stacks
pic_input = Input(shape=image_input_shape)

img_stack = Conv2D(16, (3, 3), name="convolution0", padding='same', activation=activation)(pic_input)
img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1')(img_stack)
img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2')(img_stack)
img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = Flatten()(img_stack)
img_stack = Dropout(0.2)(img_stack)

# Inject the state input
state_input = Input(shape=state_input_shape)
merged = concatenate([img_stack, state_input])

# Add a few dense layers to finish the model
merged = Dense(64, activation=activation, name='dense0')(merged)
merged = Dropout(0.2)(merged)  # to prevent overfitting
merged = Dense(10, activation=activation, name='dense2')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1, name='output')(merged)

# Optimize and compile
adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model = Model(inputs=[pic_input, state_input], outputs=merged)
model.compile(optimizer=adam, loss='mse')

"""
model.summary()
"""

# keeps learning rate from reaching too high
plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR, 'models', '{0}_model.{1}-{2}.h5'.format('model', '{epoch:02d}',
                                                                                             '{val_loss:.7f}'))
checkAndCreateDir(checkpoint_filepath)

# save model every time loss improves
checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)

# logs output of model, tracks progress
csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))

# stops training when loss stops improving, preventing overfitting
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
callbacks = [plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback, TQDMNotebookCallback()]

# train data using tensorflow-gpu
"""
with tf.device('/gpu:0'):
    history = model.fit_generator(train_generator, steps_per_epoch=num_train_examples//batch_size, epochs=500, 
    callbacks=callbacks, validation_data=eval_generator, validation_steps=num_eval_examples//batch_size, verbose=2)
"""

# automatically trains using GPU if detected using Keras
model.compile(optimizer=adam, loss='mse')
model.fit_generator(train_generator, steps_per_epoch=num_train_examples//batch_size, epochs=500, callbacks=callbacks,
                    validation_data=eval_generator, validation_steps=num_eval_examples//batch_size, verbose=2)

model.save('my_model.h5')  # saves model architecture, weights, and optimizer state
del model  # deletes existing model

model = load_model('my_model.h5')  # load model if desired

"""
[sample_batch_train_data, sample_batch_test_data] = next(train_generator)
predictions = model.predict([sample_batch_train_data[0], sample_batch_train_data[1]])
for i in range(0, 3, 1):
    draw_image_with_label(sample_batch_train_data[0][i], sample_batch_test_data[i], predictions[i])
"""
