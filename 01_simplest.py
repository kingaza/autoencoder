# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:10:02 2017

@author: hejiew
"""

#=================================================================================================
# We'll start simple, with a single fully-connected neural layer as encoder and as decoder:
#=================================================================================================

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

#-------------------------------------------------------------------------------------------------
# Let's also create a separate encoder model:
#-------------------------------------------------------------------------------------------------
# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

#-------------------------------------------------------------------------------------------------
# As well as the decoder model:
#-------------------------------------------------------------------------------------------------
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

#-------------------------------------------------------------------------------------------------
# Now let's train our autoencoder to reconstruct MNIST digits.
# First, we'll configure our model to use a per-pixel binary crossentropy loss,
# # and the Adadelta optimizer:
#-------------------------------------------------------------------------------------------------
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print("The layout of AutoEncoder Model")
autoencoder.summary()

print("The layout of encoder model")
encoder.summary()

print("The layout of dncoder model")
decoder.summary()

#-------------------------------------------------------------------------------------------------
# Let's prepare our input data. We're using MNIST digits,
# and we're discarding the labels (since we're only interested in encoding/decoding the input images).
#-------------------------------------------------------------------------------------------------
(x_train, _), (x_test, _) = mnist.load_data()

# We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# Now let's train our autoencoder for 50 epochs:
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#-------------------------------------------------------------------------------------------------
# After 50 epochs, the autoencoder seems to reach a stable train/test loss value of about 0.11.
# We can try to visualize the reconstructed inputs and the encoded representations. We will use Matplotlib.
#-------------------------------------------------------------------------------------------------
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#-------------------------------------------------------------------------------------------------
# Here's what we get. The top row is the original digits, and the bottom row is the reconstructed digits.
# We are losing quite a bit of detail with this basic approach.
#-------------------------------------------------------------------------------------------------
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()