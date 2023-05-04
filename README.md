# xAI-CycleGAN-2 - An explainability enabled evolution to CycleGAN

This project is still under development, a more thorough how-to and readme will be supplied later.
In brief summary, this new model utilizes the saliency maps of the discriminator to help the generator
learn the correct data representation, and a special merging layer with a noise masks that causes perturbations
in the data, and allows the saliency mask to affect it which further improves the learning speed.

Additionally, the model is enhanced with various features to preserve quality, sharpness and context
of the output image, enforcing only the modification of style.

