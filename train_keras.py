from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
#import keras.preprocessing.image.ImageDataGenerator
import tensorflow as tf
import keras
import os
#import cifar10

data_augmentation = False
version='2.0'
batch_size=32
num_classes=10
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_model_'+version
import matplotlib.pyplot as plt
#cifar10.maybe_download_and_extract()

#cifar10.mergeFilesIntoFile(data_dir='/opt/cifar10/cifar-10-batches-bin',outTrainFile='cifar10-Train.bin', outTestFile='cifar10-Test.bin')

#[trainLabels, trainImages] = cifar10.extractImageDataset(fileName='/opt/cifar10/cifar-10-batches-bin/cifar10-Train.bin')

#[testLabels, testImages] = cifar10.extractImageDataset(fileName='/opt/cifar10/cifar-10-batches-bin/test_batch.bin')




#x = trainImages.reshape(-1,32,32,3)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Tensorflow version:',tf.__version__)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

callback = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=15, verbose=0, mode='auto', baseline=None, restore_best_weights=True),
			ReduceLROnPlateau(patience=5, verbose=1),
            CSVLogger(filename='test_'+'log.csv'),
            ModelCheckpoint('test' + '_' + 'cifar10' + '.check',
                            save_best_only=True,
                            save_weights_only=True)]

'''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
	vertical_flip = True)
'''
model = Sequential()
#first layer
model.add(Conv2D(1024, (5,5), strides = (2,2), padding='same',data_format='channels_last',kernel_initializer='random_uniform',input_shape=(32,32,3)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#model.add(BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 	  	    moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 		  beta_constraint=None, gamma_constraint=None))

#second layer
model.add(Conv2D(256, (5,5), strides = (2,2), padding='same'))
model.add(Dropout(0.10))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#model.add(BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
#third layer
model.add(Conv2D(128, (5,5), strides = (2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#model.add(BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
#fourth layer
model.add(Conv2D(128, (5,5), strides = (2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#model.add(BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
# fully connected layers
model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(Dense(10))
model.add(Activation('softmax'))
'''
datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
'''
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
'''
#train_datagen.fit(x_train)


# Compile model
opt = keras.optimizers.adam(lr=0.001,decay = 0.0,amsgrad=True)
#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
#                    steps_per_epoch=len(x_train) / 32, epochs=100)

# 0-1 mapping fron 0 -255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
'''
# Fit the model
for e in range(100):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=batch_size):
        history = model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
'''
if data_augmentation:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format='channels_last',
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
						steps_per_epoch=len(x_train) / batch_size,
                        epochs=1000,
                        validation_data=(x_test, y_test),
			workers=4,
			callback=callback)
else:
	history = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=1000, batch_size=batch_size,shuffle=True,callback=callback)


model.load_weights('test' + '_' + 'cifar10' + '.check') # revert to the best mode

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)




scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


