# Variational Autoencoder
In this project, we will be Constructing a Variational Autoencoder, first by constructing an autoencoder to show why generating data with an autoencoder is not the best and the push towards a variational autoencoder to generate data. 


## Autoencoder:
Below is a representation of randomly selecting images from the MNSIT dataset of handwritten digits and using the autoencoder to reconstruct the image using the encoder to put the image into latent space and then using the decoder to take the latent representation and put it into the input image form:
<img width="1480" alt="Screenshot 2024-04-04 at 11 40 07 AM" src="https://github.com/karsharma10/variational_autoencoder/assets/64170090/99a501ba-d224-4f1f-9b30-3809766b882e">

Below is the representation of the latent representation of the images:

<img width="968" alt="Screenshot 2024-04-04 at 11 40 16 AM" src="https://github.com/karsharma10/variational_autoencoder/assets/64170090/98b518f7-6f57-4768-8f8f-3217b8070d7b">

A problem that is known from this is that the plot isn't symmetrical around the origin, which means some labels are represented over small areas, whereas others are over large areas. There are also gaps between the colored points (number of specific digits) so some generated images will be poor.

Thus, now that we understand why using a vanila autoencoder is not the best to generate data we will push towards a variation autoencoder to generate images instead of using a vanila autoencoder.
