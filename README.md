# sam-web
Web interface using the SAM StyleGAN to predict and interpolate between ages for a given image.

`
The task of age transformation illustrates the change of an individual's appearance over time. Accurately modeling this complex transformation over an input facial image is extremely challenging as it requires making convincing and possibly large changes to facial features and head shape, while still preserving the input identity. In this work, we present an image-to-image translation method that learns to directly encode real facial images into the latent space of a pre-trained unconditional GAN (e.g., StyleGAN) subject to a given aging shift. 
`

Model Source: https://github.com/yuval-alaluf/SAM

Reference:
```
@article{alaluf2021matter,
    author = {Alaluf, Yuval and Patashnik, Or and Cohen-Or, Daniel},
    title = {Only a Matter of Style: Age Transformation Using a Style-Based Regression Model},
    journal = {ACM Trans. Graph.},
    issue_date = {August 2021},
    volume = {40},
    number = {4},
    year = {2021},
    articleno = {45},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3450626.3459805}
}
```
