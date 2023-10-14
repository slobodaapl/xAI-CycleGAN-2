# xAI-CycleGAN-2 - An explainability enabled evolution to CycleGAN

If you like our work and want to use it in your own project, please cite us in your paper:

```
@inproceedings{sloboda2023xai,
  title={xAI-CycleGAN, a Cycle-Consistent Generative Assistive Network},
  author={Sloboda, Tibor and Hudec, Luk{\'a}{\v{s}} and Bene{\v{s}}ov{\'a}, Wanda},
  booktitle={International Conference on Computer Vision Systems},
  pages={403--411},
  year={2023},
  organization={Springer}
}
```

In brief summary, this new model utilizes the saliency maps of the discriminator to help the generator
learn the correct data representation, and a special merging layer with a noise masks that causes perturbations
in the data, and allows the saliency mask to affect it which further improves the learning speed.

Additionally, the model is enhanced with various features to preserve quality, sharpness and context
of the output image, enforcing only the modification of style.

