from tensorflow.keras import Model

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

        self._build() #will build encoder, decoder, autoencoder
    
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottle_neck = self._add_bottleneck(conv_layers) #return the whole graph of layers
        self.encoder = Model(encoder_input, bottle_neck, name = "encoder") #create a keras model pass in the input layer and output
        


