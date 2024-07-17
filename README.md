# DL_meadow_orchards
Remote Sensing course project using deep learning

## Idea
As ecologists using remote sensing, most methods are based on spectral properties. However this method has its limitation in detecting spatial patterns reliably, like meadow orchards. Deep learning could prove to be a vital tool to complement established methods.

## Methods
Performed in R using reticulate, keras3 and tensorflow.

Data input: DOP 2023 and DLM of North Rhine-Westphalia (Germany) as well as the vgg16 pretrained model.

Data augmentation: 
  - subsetting the image into tiles of the size 128x128
  - Flipping and mirroring the images to raise the number of training data

Train the model:
  - creating a new model and training it with true and false data (images)
  - combining that model with a pre trained model (vgg16), using the first 15 layer for edge and structure detection


## Results
For now the model prdicts everything as 0, meaning no meadow orchard. This is due to a disbalnace of training data (72.000 F to 71 T)
