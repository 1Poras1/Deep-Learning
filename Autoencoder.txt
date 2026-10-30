Autoencoder in points

1. Structure

Input: the image, flattened (28×28 = 784 pixels).
Encoder: layers get smaller (784 → 256 → 64), which compresses the data.
Latent code z: the bottleneck, a tiny vector (for example 2 to 32 numbers).
Decoder: layers get bigger again (64 → 256 → 784), which rebuilds the image.
Output: a reconstructed image with the same size as the input.

2. Training

The target is the input itself, so no labels are needed (unsupervised).
Loss = difference between input and output (usually MSE).
Backpropagation updates the encoder and decoder together (the dashed arrow in the diagram).

3. Dimensionality reduction

The narrow bottleneck forces the network to keep only the most useful features and drop redundancy and noise.
After training, use the encoder alone to turn each image into a small z.
With non-linear activations it can capture patterns that PCA cannot.

4. Image reconstruction

The decoder maps z back to pixels.
A smaller z gives more compression but blurrier images.
A larger z gives sharper images but less compression.

5. Applications

Dimensionality reduction and 2-D visualization
Image compression
Denoising (noisy input, clean target)
Anomaly detection (a high reconstruction error flags something unusual)
Feature learning and pretraining
Inpainting and super-resolution

6. Limitations

Reconstructions are blurry.
It works best on data similar to the training data.
Plain autoencoders are poor generators of new images, which is why VAEs exist.
