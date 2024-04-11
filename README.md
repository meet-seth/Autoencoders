# Image Compression using Autoencoders

## Info

The aim of this repository is to create a training pipeline for the use of **Autoencoders** for **Image Compression** tasks. This pipeline allows user to simply configure the model using ***model.json*** file, defining custom dataset directory and specify hyperparameters to train an *Autoencoder* model. Functionality to export individual encoder and decoder models OR full model (including encoder and decoder for testing purposes) has been provided in ***exporter.py***. Each file has a bash script with the same name to demonstrate the use of each file using command line. However, the structure is created in such a way that you can integrate these classes in your own python workflow. 

The experiments has been done on **MNIST** Dataset and all the result metrics, model graphs, etc. are with respect to the same although support has been provided for a few of datasets available on `tensorflow-datasets` along with the use of *custom dataset*. 

The entire pipeline is developed in `tensorflow==2.14.1`. Therefore, for best results, it is preferrable to use `tensorflow==2.14.1`.

Please feel free to use this code and modify it according to your own needs and don't forget to cite this repo. :grinning:

## Model Architecture

Model architecture for autoencdoer consists of an **Encoder Model** and a **Decoder Model**. Both these models are inside the model **Generator** which is further used inside the model **ImageCompressor**. The reason for this is to allow **Discriminator** model which has **NOT** been used for **MNIST** dataset. To use a **Discriminator** one can simply create a new class `Discriminator` which subclasses `tf.keras.models.Model` while tweaking loss function dictionary and `train_step` and `test_step` functions. Another reason for keeping `ImageCompressor` and `Generator` models different is to keep the code flow smooth. To modify or add models, make changes to [`ModelArchitecture.py`](/architectures/ModelArchitecture.py)

The image below shows the model architecture along with input and outptut layers for **MNIST** dataste.
![Model Architecture](/imgs_github/model_arch.png)

## Loss Function

Loss Function used for our approach is a combination of SSIM loss and L1 loss or Mean Absolute Loss. We aim to **maximize SSIM** metric and **minimize L1** metric. 

Formula for SSIM is given by:

$$
 SSIM(x,y) = (2*{\mu}_x*{\mu}_y + c_1)*(2*{\sigma}_{xy} + c_2) / ({\mu}^2_x + {\mu}^2_y + c_1)*({\sigma}^2_x + {\sigma}^2_y + c_2)
$$

with:

    μ x {\displaystyle \mu _{x}} the pixel sample mean of x {\displaystyle x};
    μ y {\displaystyle \mu _{y}} the pixel sample mean of y {\displaystyle y};
    σ x 2 {\displaystyle \sigma _{x}^{2}} the variance of x {\displaystyle x};
    σ y 2 {\displaystyle \sigma _{y}^{2}} the variance of y {\displaystyle y};
    σ x y {\displaystyle \sigma _{xy}} the covariance of x {\displaystyle x} and y {\displaystyle y};
    c 1 = ( k 1 L ) 2 {\displaystyle c_{1}=(k_{1}L)^{2}}, c 2 = ( k 2 L ) 2 {\displaystyle c_{2}=(k_{2}L)^{2}} two variables to stabilize the division with weak denominator;
    L {\displaystyle L} the dynamic range of the pixel-values (typically this is 2 # b i t s   p e r   p i x e l − 1 {\displaystyle 2^{\#bits\ per\ pixel}-1});
    k 1 = 0.01 {\displaystyle k_{1}=0.01} and k 2 = 0.03 {\displaystyle k_{2}=0.03} by default.


