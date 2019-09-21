from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # this is input layer
        # CONV => RELU => POOL
        model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape=inputShape))
        # 32 features maps are generated using 32 3x3 filters
        # value of matrix in each filter is random at first
        # filter matrix value change when weight and biased are updated
        model.add(Activation("relu"))
        # relu removes negative value from features map keeping positive value as it is eg [[-1,22],[3,-5]]=[[0,22],[3,0]]
        model.add(BatchNormalization(axis=chanDim))
        # Normalization means 2 things:
        # Putting the data on the same scale (Scaling)
        # Balancing the data around a point (Centering)
        # Now you could only do one without the other, but both bring their own specific benefits.
        # Scaling improves convergence speed and accuracy
        # Centering fights vanishing and exploding gradients, while probably also increasing convergence speed and accuracy

        model.add(MaxPooling2D(pool_size=(2, 2)))
        # after maxpooling size of image become25x25 from 50x50
        model.add(Dropout(0.25))

        # (CONV => RELU => POOL) * 2
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # this is second convolutional layer
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # after maxpooling size of image become 12x12 from 25x25
        model.add(Dropout(0.25))

        # (CONV => RELU => POOL) * 3
        # this is third convolutional layer
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # this is fourth convolutional layer
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # this is fifth convolutional layer
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # after maxpooling size of image become 6x6 from 12x12
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        # change n-dimensional array into 1-dimensional eg [[1,2],[4,5]]=[1,2,4,5]
        # first hidden layer with 256 neurons
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # second hidden layer with 128 neurons
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        #  third layer with 16 neurons
        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # this is o/p layer with 2 neurons
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
