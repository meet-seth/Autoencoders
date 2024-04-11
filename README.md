# Image Compression using Autoencoders

## Info

The aim of this repository is to create a training pipeline for the use of **Autoencoders** for **Image Compression** tasks. This pipeline allows user to simply configure the model using ***model.json*** file, defining custom dataset directory and specify hyperparameters to train an *Autoencoder* model. Functionality to export individual encoder and decoder models OR full model (including encoder and decoder for testing purposes) has been provided in ***exporter.py***. Each file has a bash script with the same name to demonstrate the use of each file using command line. However, the structure is created in such a way that you can integrate these classes in your own python workflow. 

The experiments has been done on **MNIST** Dataset and all the result metrics, model graphs, etc. are with respect to the same although support has been provided for a few of datasets available on `tensorflow-datasets` along with the use of *custom dataset*. 

> [!TIP]
> The entire pipeline is developed in `tensorflow==2.14.1`. Therefore, for best results, it is preferrable to use `tensorflow==2.14.1`.

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

## Usage
> [!IMPORTANT]
> First of all, to use the pipeline, we need to create a model.json file which would contain the architecture of the model. Take a look at [model.json](/models/model.json) for referrence.
> Do provide all the arguments neccessary for the Layer to be built required by tensorflow. Refer [tensorflow docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers) on what arguments are neccessary for a layer.

After creating the model.json file, we can perform training, validation and inference easily. Stick to the end for there is a fun GUI which demonstrates the working of the model.

### Training

To perform training we need to run [train.py](/train.py). The following are the arguments:

| Arguments | Usage |
| --- | --- |
|`--model` | This is the path to model.json file we just created. Look at [ModelBuilder.py](/builders/ModelBuilder.py) to see how a model is generated from json file. |
| `--tfds` | This argument is to specify if the dataset is available on tensorflow datasets. For now only a few datasets are supported by the code due to different split ratios and different preprocessing required for each, but it is very easy to add support for new datasets. Look at [DatasetBuilder.py](/builders/DatasetBuilder.py) on how it works. |
| `--dataset` | Name of the dataset if `--tfds` is set to `True` otherwise pass in the path of images to your custom dataset directory. See [DatasetBuilder.py](/builders/DatasetBuilder.py) to see how custom dataset directory path is treated. |
| `--log_dir` | Tensorboard is used to log information during training and validation. To pass in a specific directory to store the logs, use this argument. Otherwise the logs will be stored by default to `./log/tensorboard/`. See [constants.py](/constants/constants.py) for more details on default paramerters. |
| `--batch_size` | Batch Size Argument to train the model with. Default is set to 32 |
| `--learning_rate` | Learning rate argument for the optimizer (here Adam). Deafult is set to 1e-3 |
| `--latent_dims` | The size of the latent compressed dimensions. This is the size of `compressed_representation_layer` in Model Architecture. Refer the above image on Model Architecture. Default is set to 50 |
| `--epochs` | Number of epochs to run training loop for. Default is set to 15. |
| `--load_from_ckpt_path` | In case you are continuing training or doing transfer learning, pass in the path of model checkpoints to load weights from the checkpoint. Please note that model.json file is still required to build the model. Defaults to `None` |
| `--checkpoint_filepath` | Path to save model checkpoints while training. This is a path passed to `tf.keras.callbacks.Checkpoint()`. Default is set to `./checkpoints/model.ckpt` |
| `--verbosity` | One of 0,1,2 where 0 - silent, 1 - progress bar, 2 - one line per epoch. |

I have implemented a [`trainer.sh`](/trainer.sh) shell script to run the above file. It is created for running the python file easily. 

> [!NOTE]
> TPU is not supported by the code at this moment. But if someone if using TPU, do modify the code in [DatasetBuilder.py](/builders/DatasetBuilder.py) in build, to load tfds dataset with `try_gcs=True` if the dataset is available in `tfds`.

### Validation

To perform validation we need to run [validation.py](/validation.py). The following are the arguments:

| Arguments | Usage |
| --- | --- |
|`--model` | This is the path to model.json file we just created. Look at [ModelBuilder.py](/builders/ModelBuilder.py) to see how a model is generated from json file. |
| `--tfds` | This argument is to specify if the dataset is available on tensorflow datasets. For now only a few datasets are supported by the code due to different split ratios and different preprocessing required for each, but it is very easy to add support for new datasets. Look at [DatasetBuilder.py](/builders/DatasetBuilder.py) on how it works. |
| `--dataset` | Name of the dataset if `--tfds` is set to `True` otherwise pass in the path of images to your custom dataset directory. See [DatasetBuilder.py](/builders/DatasetBuilder.py) to see how custom dataset directory path is treated. |
| `--log_dir` | Tensorboard is used to log information during training and validation. To pass in a specific directory to store the logs, use this argument. Otherwise the logs will be stored by default to `./log/tensorboard/`. See [constants.py](/constants/constants.py) for more details on default paramerters. |
| `--batch_size` | Batch Size Argument to train the model with. Default is set to 32 |
| `--latent_dims` | The size of the latent compressed dimensions. This is the size of `compressed_representation_layer` in Model Architecture. Refer the above image on Model Architecture. Default is set to 50. This is a required argument to be passed in again to build the model. |
| `--ckpt_path` | The file path saved checkpoints to load the weights in the built model. |

I have implemented a [`validator.sh`](/validator.sh) shell script to run the above file. It is created for running the python file easily.