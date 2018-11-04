from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import tensorflow as tf
import keras
import os
#import cifar10

version='1.0'
batch_size=128
num_classes=10
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_model_'+version
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

model = Sequential()
#first layer

model.add(Conv2D(256, (3,3), strides = (2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

#second layer
model.add(Conv2D(128, (3,3), strides = (2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

#third layer
model.add(Conv2D(64, (3,3), strides = (2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

#fourth layer
model.add(Conv2D(64, (3,3), strides = (2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# fully connected layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.90))
model.add(Dense(10))
model.add(Activation('softmax'))

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
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 0-1 mapping fron 0 -255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=100, batch_size=batch_size,shuffle=True)


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


