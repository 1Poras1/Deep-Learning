Generator:

Random Noise (100 numbers)
↓
Dense Layer
↓
Reshape (7 x 7 x 128)
↓
Conv2DTranspose
↓
Generated Image (28 x 28 x 1)

Discriminator:
Input Image (28 x 28 x 1)
↓
Conv2D
↓
LeakyReLU
↓
Conv2D
↓
LeakyReLU
↓
Flatten
↓
Dense Layer
↓
Output (Real or Fake)





Links:- https://www.tensorflow.org/tutorials/generative/dcgan

        https://docs.langchain.com/oss/python/langchain/overview

        https://www.slideshare.net/slideshow/deep-learning-az-convolutional-neural-networks-cnn-module-2/110293079
