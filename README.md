# DL_meadow_orchards
Remote Sensing course project using deep learning

## Idea
As ecologists using remote sensing, most methods are based on spectral properties. However this method has its limitation in detecting spatial patterns reliably, like meadow orchards. Deep learning could prove to be a vital tool to complement established methods.

## Methods
Performed in R using reticulate, keras3 and tensorflow.

Data input: DOP 2023 and DLM of North Rhine-Westphalia (Germany) as well as the vgg16 pretrained U-Net model.

11 AOI´s including a traditional orchard. Create a mask for every AOI indicating T/F.

Data augmentation: 
  - subsetting the image into tiles of the size 448x448
  - Flipping and mirroring the images to raise the number of training data

Train the model:
  - pre trained model (vgg16), using the first 15 layer for edge and structure detection


## Results

Model gives a prediction per tile (448x448 subset) that can be reassambled to a map displaying the probability for the presence of a traditional orchard.
Workflow is operating as hoped, the output however does not show reliable results as no spatial patterns can be recognised.
