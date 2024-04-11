# Image Compression using Autoencoders

## Info

The aim of this repository is to create a training pipeline for the use of **Autoencoders** for **Image Compression** tasks. This pipeline allows user to simply configure the model using ***model.json*** file, defining custom dataset directory and specify hyperparameters to train an *Autoencoder* model. Functionality to export individual encoder and decoder models OR full model (including encoder and decoder for testing purposes) has been provided in ***exporter.py***. Each file has a bash script with the same name to demonstrate the use of each file using command line. However, the structure is created in such a way that you can integrate these classes in your own python workflow. 

The experiments has been done on **MNIST** Dataset and all the result metrics, model graphs, etc. are with respect to the same although support has been provided for a few of datasets available on `tensorflow-datasets` along with the use of *custom dataset*. 

The entire pipeline is developed in `tensorflow==2.14.1`. Therefore, for best results, it is preferrable to use `tensorflow==2.14.1`.

Please feel free to use this code and modify it according to your own needs and don't forget to cite this repo. :grinning:

## Model Architecture

Model architecture for autoencdoer consists of an **Encoder Model** and a **Decoder Model**. Both these models are inside the model **Generator** which is further used inside the model **ImageCompressor**. The reason for this is to allow **Discriminator** model which has **NOT** been used for **MNIST** dataset. To use a **Discriminator** one can simply create a new class `Discriminator` which subclasses `tf.keras.models.Model` while tweaking loss function dictionary and `train_step` and `test_step` functions. Another reason for keeping `ImageCompressor` and `Generator` models different is to keep the code flow smooth. To modify or add models, make changes to [ModelArchitecture.py](/architectures/ModelArchitecture.py)

The image below shows the model architecture along with input and outptut layers for **MNIST** dataste.
![Model Architecture](/imgs_github/model_arch.png)

## Loss Function

Loss Function used for our approach is a combination of SSIM loss and L1 loss or Mean Absolute Loss. We aim to **maximize SSIM** metric and **minimize L1** metric. 

Formula for **SSIM** is given by:

```math
\displaylines{
 SSIM(x,y) = \frac{(2*{\mu}_x*{\mu}_y + c_1)*(2*{\sigma}_{xy} + c_2)} {({\mu}^2_x + {\mu}^2_y + c_1)*({\sigma}^2_x + {\sigma}^2_y + c_2)} 
}
```
with :
 - $\mu_x$ = pixel mean of x;
 - $\mu_y$ = pixel mean of y;
 - $\sigma_x^2$ = variance of x;
 - $\sigma_y^2$ = variance of y;
 - $\sigma_{xy}$ = covariance of x and y;
 - $c_1 = (k_1*L)^2, c_2 = (k_2*:)^2$ two variables to stabilize division with weak denominator;
 - $L$ = dynamic range of pixel values
 - $k_1 = 0.01, k_2=0.03$ by default 

Formula for **Mean Absolute Error** is given by:
```math
MAE(x,y) = \frac{1}{N} * \sum_{i=0}^{m-1}\sum_{j=0}^{n-1}{|x_(i,j) - y_(i,j)|}
```
where:
 - $x_i$ is the pixel at the $i^th$ location in image $x$
 - $y_i$ is the pixel at the $i^th$ location in image $y$
 - $N = m*n$
 - $m,n$ are the height and width of the image

Hence out final loss becomes:
```math
Loss = \alpha * (1-SSIM(x,y)) + (1 - \alpha) * MAE(x,y)
```
where $\alpha$ is the weight factor.

For customizations to loss function modify the code in [LossArchitecture.py](/architectures/LossArchitecture.py)

## Metrics
We use two metrics, SSIM metric and PSNR ratio to determine the performance of the model. both are availabe in tf.image ready to use. 

Formula for SSIM is the same as above.
Formula for PSNR is given by:

MSE of channel x: 
```math
MSE_x = \frac{1}{N} * \sum_{i=0}^{m-1} \sum_{j=0}{n-1} [x_(i,j) - y_(i,j)]^2 , N = m*n
```

Total MSE:
```math
MSE_t = MSE_R + MSE_G + MSE_B
```

Calculate PSNR

```math
PSNR = 10 * \log_{10} \huge( \small\frac{MAX_I^2}{MSE_t} \huge) 
```
where $MAX_I$ is the maximum value an image can hold. In case of uint8 its 255 while for normalized image its 1. 

To modify or add new metrics, checkout the file [MetricsArchitecture.py](/architectures/MetricsArchitecture.py)

## Results

The following graphs show the performance of our ImageCompressor model on MNIST dataset:
![EPOCHS_VS_SSIM](/imgs_github/eopchs_vs_ssim.png) ![EPOCHS_VS_PSNR](/imgs_github/eopchs_vs_psnr.png) ![EPOCHS_VS_LOSS](/imgs_github/epochs_vs_loss.png)

The following are example regenerated images :
![Example](/imgs_github/regen_images.png)

The left column of images are input or real images while the right columns are the output or regenerated images by the decoder.
