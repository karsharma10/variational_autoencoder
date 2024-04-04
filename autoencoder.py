from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os
import pickle
import numpy as np

class Autoencoder:
    '''
    Autoencoder represents a deep convolutional autoencoer architecture with mirroed encoder and decoder components.
    '''

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels 
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None #tensor flow encoder
        self.decoder = None #tensor flow decoder
        self.model = None #tensor flow model

        self._num_conv_layers = len(conv_filters) #number of convolution filters
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build() #will build encoder, decoder, autoencoder
    
    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
    
    def _create_folder_if_it_doesnt_exist(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5") #h5 is keras format for storing weights
        self.model.save_weights(save_path)


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001): #compile keras model
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer,loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,x_train, batch_size= batch_size, epochs = num_epochs, shuffle = True) #essentially we want the input and output to be xtrain, and data will be shufled before training.


    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
    
    def _build_autoencoder(self):
        model_input = self._model_input
        mode_output = self.decoder(self.encoder(model_input)) #output of the whole autoencoder
        self.model = Model(model_input, mode_output, name="autoencoder")
    
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")
    
    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name = "decoder_input")

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottle_neck = self._add_bottleneck(conv_layers) #return the whole graph of layers
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottle_neck, name = "encoder") #create a keras model pass in the input layer and output

    def _add_encoder_input(self):
        return Input(shape = self.input_shape, name = "encoder_input")
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshape_layer
    
    def _add_conv_transpose_layers(self, x):
        '''
        Add conv tranpose blocks. Loop through all the conv layers in reverse order and stop at the first layer.
        '''

        for layer_index in reversed(range(1,self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index,x)

        return x
    
    def _add_conv_transpose_layer(self,layer_index,x):
        layer_num = self._num_conv_layers - layer_index
        conv_tranpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"decoder_conv_tranpose_layer{layer_num}"
        )

        x = conv_tranpose_layer(x)
        x = ReLU(name = f"decoder_relu{layer_num}")(x)
        x = BatchNormalization(name = f"decoder_bn_{layer_num}")(x)

        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters = 1,
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = "same",
            name = f"decoder_conv_tranpose_layer{self._num_conv_layers}"
        )

        x = conv_transpose_layer(x)

        output_layer = Activation("sigmoid", name = "sigmoid_layer")(x)
        return output_layer

    def _add_conv_layers(self, encoder_input):
        """
        Creates all conlutional blocks in encoder.
        """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        adds a convolution block to a graph of layers, consisting of conv 2d  + ReLu + batch_normalization layer
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding="same",
            name = f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name = f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name = f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        '''
        Flatten the data and add bottleneck (Dense layer).
        '''
        self._shape_before_bottleneck = K.int_shape(x)[1:] #shape of the data before we flatten
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)

        return x
    
if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape= (28,28,1),
        conv_filters=(32, 64, 64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=2
    )

    autoencoder.summary()
    
