from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras import backend as K

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

        self._build() #will build encoder, decoder, autoencoder
    
    def summary(self):
        self.encoder.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
    
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
        self.encoder = Model(encoder_input, bottle_neck, name = "encoder") #create a keras model pass in the input layer and output

    def _add_encoder_input(self):
        return Input(shape = self.input_shape, name = "encoder_input")
    
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
    
