# Setup on Win11 Laptop
#### Packages ####
library(terra)
library(caret)
library(CAST)
library(sf)
library(stars)
library(rsample)
library(tidyverse)
library(tfdatasets)

library(jpeg)
library(ggplot2)
library(mapview)
library(purrr)
library(magick)
library(surveytoolbox) # timed_fn() to create timestamps for files

library(neuralnet)
library(stringr)
library(pbapply)

#### Setup ####
library(reticulate)
use_virtualenv("C:/Users/esanc/Documents/.virtualenvs/r-tensorflow", required = TRUE)

library(keras3)
library(tensorflow)

tf$version # verify installation
reticulate::py_config()
tensorflow::tf_config()


#### Functions ####

dl_subsets <- function(inputrst, targetsize, targetdir, targetname="", img_info_only = FALSE, is_mask = FALSE){
  require(jpeg)
  require(raster)
  
  #determine next number of quadrats in x and y direction, by simple rounding
  targetsizeX <- targetsize[1]
  targetsizeY <- targetsize[2]
  inputX <- ncol(inputrst)
  inputY <- nrow(inputrst)
  
  #determine dimensions of raster so that 
  #it can be split by whole number of subsets (by shrinking it)
  while(inputX%%targetsizeX!=0){
    inputX = inputX-1  
  }
  while(inputY%%targetsizeY!=0){
    inputY = inputY-1    
  }
  
  #determine difference
  diffX <- ncol(inputrst)-inputX
  diffY <- nrow(inputrst)-inputY
  
  #determine new dimensions of raster and crop, 
  #cutting evenly on all sides if possible
  newXmin <- floor(diffX/2)
  newXmax <- ncol(inputrst)-ceiling(diffX/2)-1
  newYmin <- floor(diffY/2)
  newYmax <- nrow(inputrst)-ceiling(diffY/2)-1
  rst_cropped <- suppressMessages(crop(inputrst, raster::extent(inputrst,newYmin,newYmax,newXmin,newXmax)))
  #writeRaster(rst_cropped,filename = target_dir_crop)
  
  #return (list(ssizeX = ssizeX, ssizeY = ssizeY, nsx = nsx, nsy =nsy))
  agg <- suppressMessages(aggregate(rst_cropped[[1]],c(targetsizeX,targetsizeY)))
  agg[]    <- suppressMessages(1:ncell(agg))
  agg_poly <- suppressMessages(rasterToPolygons(agg))
  names(agg_poly) <- "polis"
  
  pb <- txtProgressBar(min = 0, max = ncell(agg), style = 3)
  for(i in 1:ncell(agg)) {
    
    # rasterOptions(tmpdir=tmpdir)
    setTxtProgressBar(pb, i)
    e1  <- raster::extent(agg_poly[agg_poly$polis==i,])
    
    subs <- suppressMessages(crop(rst_cropped,e1))
    #rescale to 0-1, for jpeg export
    if(is_mask==FALSE){
      
      subs <- suppressMessages((subs-cellStats(subs,"min"))/(cellStats(subs,"max")-cellStats(subs,"min")))
    } 
    #write jpg
    
    
    writeJPEG(as.array(subs),target = paste0(targetdir,targetname,i,".jpg"),quality = 1)
    
    #writeRaster(subs,filename=paste0(targetdir,"SplitRas_",i,".tif"),overwrite=TRUE) 
    #return(c(raster::extent(rst_cropped),crs(rst_cropped)))
  }
  close(pb)
  #img_info <- list("tiles_rows"=nrow(rst_cropped)/targetsizeY, "tiles_cols"=ncol(rst_cropped)/targetsizeX,"crs"= crs(rst_cropped),"extent"=extent(rst_cropped))
  #writeRaster(rst_cropped,filename = paste0(targetdir,"input_rst_cropped.tif"))
  rm(subs,agg,agg_poly)
  gc()
  return(rst_cropped)
  
}

rebuild_img <- function(pred_subsets,out_path,target_rst){
  require(raster)
  require(gdalUtils)
  require(stars)
  
  
  subset_pixels_x <- ncol(pred_subsets[1,,,])
  subset_pixels_y <- nrow(pred_subsets[1,,,])
  tiles_rows <- nrow(target_rst)/subset_pixels_y
  tiles_cols <- ncol(target_rst)/subset_pixels_x
  
  # load target image to determine dimensions
  target_stars <- st_as_stars(target_rst,proxy=F)
  #prepare subfolder for output
  result_folder <- paste0(out_path,"out")
  if(dir.exists(result_folder)){
    unlink(result_folder,recursive = T)
  }
  dir.create(path = result_folder)
  
  #for each tile, create a stars from corresponding predictions, 
  #assign dimensions using original/target image, and save as tif: 
  for (crow in 1:tiles_rows){
    for (ccol in 1:tiles_cols){
      i <- (crow-1)*tiles_cols + (ccol-1) +1 
      
      dimx <- c(((ccol-1)*subset_pixels_x+1),(ccol*subset_pixels_x))
      dimy <- c(((crow-1)*subset_pixels_y+1),(crow*subset_pixels_y))
      cstars <- st_as_stars(t(pred_subsets[i,,,1]))
      attr(cstars,"dimensions")[[2]]$delta=-1
      #set dimensions using original raster
      st_dimensions(cstars) <- st_dimensions(target_stars[,dimx[1]:dimx[2],dimy[1]:dimy[2]])[1:2]
      
      write_stars(cstars,dsn = paste0(result_folder,"/_out_",i,".tif")) 
    }
  }
  
  # starstiles <- as.vector(list.files(result_folder,full.names = T),mode = "character")
  # gdalbuildvrt(starstiles,paste0(result_folder,"/mosaic.vrt"))
  # gdalwarp(paste0(result_folder,"/mosaic.vrt"), paste0(result_folder,"/mosaic.tif"))
  
  ff <- list.files(path = result_folder, pattern = '//.tif$', full.names = TRUE)
  ff <- mixedsort(sort(ff))
  v <- vrt(ff, "dem.vrt")
  writeRaster(v, "out.tif")
}

spectral_augmentation <- function(img) {
  img <- tf$image$random_brightness(img, max_delta = 0.3) 
  img <- tf$image$random_contrast(img, lower = 0.8, upper = 1.2)
  img <- tf$image$random_saturation(img, lower = 0.8, upper = 1.2) 
  # make sure we still are between 0 and 1
  img <- tf$clip_by_value(img,0, 1) 
}

plot_layer_activations <- function(img_path, model, activations_layers,channels){
  
  
  model_input_size <- c(model$input_shape[[2]], model$input_shape[[3]]) 
  
  #preprocess image for the model
  img <- image_load(img_path, target_size =  model_input_size) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, model_input_size[1], model_input_size[2], 3)) %>%
    imagenet_preprocess_input()
  
  layer_outputs <- lapply(model$layers[activations_layers], function(layer) layer$output)
  activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
  activations <- predict(activation_model,img)
  if(!is.list(activations)){
    activations <- list(activations)
  }
  
  #function for plotting one channel of a layer, adopted from: Chollet (2018): "Deep learning with R"
  plot_channel <- function(channel,layer_name,channel_name) {
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(channel), axes = FALSE, asp = 1,
          col = terrain.colors(12),main=paste("layer:",layer_name,"channel:",channel_name))
  }
  
  for (i in 1:length(activations)) {
    layer_activation <- activations[[i]]
    layer_name <- model$layers[[activations_layers[i]]]$name
    n_features <- dim(layer_activation)[[4]]
    for (c in channels){
      
      channel_image <- layer_activation[1,,,c]
      plot_channel(channel_image,layer_name,c)
      
    }
  } 
  
}

#adapted from: https://blogs.rstudio.com/ai/posts/2019-08-23-unet/ (accessed 2020-08-12)
dl_prepare_data <- function(files=NULL, train, predict=FALSE, subsets_path=NULL, model_input_shape = c(448,448), batch_size = 10L) {
  
  if (!predict){
    
    #function for random change of saturation,brightness and hue, 
    #will be used as part of the augmentation
    spectral_augmentation <- function(img) {
      img <- tf$image$random_brightness(img, max_delta = 0.3)
      img <- tf$image$random_contrast(img, lower = 0.8, upper = 1.1)
      img <- tf$image$random_saturation(img, lower = 0.8, upper = 1.1)
      # make sure we still are between 0 and 1
      img <- tf$clip_by_value(img, 0, 1)
    }
    
    
    #create a tf_dataset from the input data.frame 
    #right now still containing only paths to images 
    dataset <- tensor_slices_dataset(files)
    
    #use dataset_map to apply function on each record of the dataset 
    #(each record being a list with two items: img and mask), the 
    #function is list_modify, which modifies the list items
    #'img' and 'mask' by using the results of applying decode_jpg on the img and the mask   
    #-> i.e. jpgs are loaded and placed where the paths to the files were (for each record in dataset)
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x,img = tf$image$decode_jpeg(tf$io$read_file(.x$img)),
                    mask = tf$image$decode_jpeg(tf$io$read_file(.x$mask)))) 
    
    #convert to float32:
    #for each record in dataset, both its list items are modyfied 
    #by the result of applying convert_image_dtype to them
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x, img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
                    mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float32))) 
    
    #resize:
    #for each record in dataset, both its list items are modified 
    #by the results of applying resize to them 
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x, img = tf$image$resize(.x$img, size = shape(model_input_shape[1], model_input_shape[2])),
                    mask = tf$image$resize(.x$mask, size = shape(model_input_shape[1], model_input_shape[2]))))
    
    
    # data augmentation performed on training set only
    if (train) {
      
      #augmentation 1: flip left right, including random change of 
      #saturation, brightness and contrast
      
      #for each record in dataset, only the img item is modified by the result 
      #of applying spectral_augmentation to it
      augmentation <- 
        dataset_map(dataset, function(.x) 
          list_modify(.x, img = spectral_augmentation(.x$img)))
      
      #...as opposed to this, flipping is applied to img and mask of each record
      augmentation <- 
        dataset_map(augmentation, function(.x) 
          list_modify(.x, img = tf$image$flip_left_right(.x$img),
                      mask = tf$image$flip_left_right(.x$mask)))
      
      dataset_augmented <- dataset_concatenate(dataset,augmentation)
      
      #augmentation 2: flip up down, 
      #including random change of saturation, brightness and contrast
      augmentation <- 
        dataset_map(dataset, function(.x) 
          list_modify(.x, img = spectral_augmentation(.x$img)))
      
      augmentation <- 
        dataset_map(augmentation, function(.x) 
          list_modify(.x, img = tf$image$flip_up_down(.x$img),
                      mask = tf$image$flip_up_down(.x$mask)))
      
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      
      #augmentation 3: flip left right AND up down, 
      #including random change of saturation, brightness and contrast
      
      augmentation <- 
        dataset_map(dataset, function(.x) 
          list_modify(.x, img = spectral_augmentation(.x$img)))
      
      augmentation <- 
        dataset_map(augmentation, function(.x) 
          list_modify(.x, img = tf$image$flip_left_right(.x$img),
                      mask = tf$image$flip_left_right(.x$mask)))
      
      augmentation <- 
        dataset_map(augmentation, function(.x) 
          list_modify(.x, img = tf$image$flip_up_down(.x$img),
                      mask = tf$image$flip_up_down(.x$mask)))
      
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      
    }
    
    # shuffling on training set only
    if (train) {
      dataset <- dataset_shuffle(dataset_augmented, buffer_size = batch_size*128)
    }
    
    # train in batches; batch size might need to be adapted depending on
    # available memory
    dataset <- dataset_batch(dataset, batch_size)
    
    # output needs to be unnamed
    dataset <-  dataset_map(dataset, unname) 
    
  }else{
    #make sure subsets are read in in correct order 
    #so that they can later be reassembled correctly
    #needs files to be named accordingly (only number)
    o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(subsets_path)))))
    subset_list <- list.files(subsets_path, full.names = T)[o]
    
    dataset <- tensor_slices_dataset(subset_list)
    
    dataset <- 
      dataset_map(dataset, function(.x) 
        tf$image$decode_jpeg(tf$io$read_file(.x))) 
    
    dataset <- 
      dataset_map(dataset, function(.x) 
        tf$image$convert_image_dtype(.x, dtype = tf$float32)) 
    
    dataset <- 
      dataset_map(dataset, function(.x) 
        tf$image$resize(.x, size = shape(model_input_shape[1], model_input_shape[2]))) 
    
    dataset <- dataset_batch(dataset, batch_size)
    dataset <-  dataset_map(dataset, unname)
    
  }
  
}


#### Data preperation #####

### create DOP 
f <- list.files("data/tiles/uedem", pattern = "*.jp2$", full.names = T) # list dop tiles to read later

dop <- list()
for (i in 1:length(f)) { # read tiles into a list
  
  dop[[length(dop)+1]] <- rast(f[i])
}
uedem <- do.call(merge,dop) # uedem DOP
writeRaster(uedem, "data/tiles/uedem.tif", overwrite=FALSE)


#### Crop tiles and mask
# after QGis

size <- c(448,448)

dir_img1 <- "data/training/subsets/"
img1 <- stack("data/img_cropped_uedem.tif")
dl_subsets(img1, size, dir_img1, targetname="1111") # create the subsets of the image

dir_mask1 <- "data/training/masks/"
mask1 <- stack("data/mask_uedem.tif")
dl_subsets(mask1, size, dir_mask1, targetname="") # create the subsets of the mask


### Rename images 

library(stringr)

directory <- "data/training/subsets/"
directory <- "data/training/masks/"

files <- list.files(directory, full.names = TRUE)
# Function to rename files
rename_files <- function(file) {
  # Extract the file extension
  extension <- tools::file_ext(file)
  
  # Extract the base name (without extension)
  basename <- tools::file_path_sans_ext(basename(file))
  
  # Extract the number from the base name
  new_basename <- str_extract(basename, "//d+$")
  
  # Create the new file name
  new_name <- paste0(new_basename, ".", extension)
  
  # Create the full path for the new name
  new_path <- file.path(dirname(file), new_name)
  
  # Rename the file
  file.rename(file, new_path)
}

sapply(files, rename_files) # apply function








#### Augmentation ####

files <- data.frame( #get paths
  img = list.files("data/training/subsets/", full.names = TRUE, pattern = "*.jpg"),
  mask = list.files("data/training/masks/", full.names = TRUE, pattern = "*.jpg"))

# split the data into training and validation datasets. 
files <- initial_split(files, prop = 0.8)

# prepare data for training
training_dataset <- dl_prepare_data(training(files),train = TRUE,model_input_shape = c(128,128),batch_size = 10L)
validation_dataset <- dl_prepare_data(testing(files),train = FALSE,model_input_shape = c(128,128),batch_size = 10L)

# get all tensors through the python iterator
training_tensors <- training_dataset%>%as_iterator()%>%iterate()

#how many tensors?
length(training_tensors)




#### U-Net ####

vgg16_feat_extr <- application_vgg16(include_top = F,input_shape = c(128,128,3),weights = "imagenet")
#freeze weights
freeze_weights(vgg16_feat_extr)
#use only layers 1 to 15
pretrained_model <- keras_model_sequential(vgg16_feat_extr$layers[1:15])


# add flatten and dense layers for classification 
# -> these dense layers are going to be trained on our data only
pretrained_model <- layer_flatten(pretrained_model)
pretrained_model <- layer_dense(pretrained_model,units = 256,activation = "relu")
pretrained_model <- layer_dense(pretrained_model,units = 1,activation = "sigmoid")

pretrained_model



compile(
  pretrained_model,
  optimizer = optimizer_rmsprop(learning_rate = 1e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#### test to predict


# get all file paths of the images of our test area
subset_list <- list.files(path="data/training/subsets/", full.names = T) # all images in one folder for the prediction

dataset <- tensor_slices_dataset(subset_list)
dataset <- dataset_map(dataset, function(.x) 
  tf$image$decode_jpeg(tf$io$read_file(.x))) 
dataset <- dataset_map(dataset, function(.x) 
  tf$image$convert_image_dtype(.x, dtype = tf$float32)) 
dataset <- dataset_map(dataset, function(.x) 
  tf$image$resize(.x, size = shape(128, 128))) 
dataset <- dataset_batch(dataset, 10L)
dataset <-  dataset_map(dataset, unname)

# save_model_hdf5(first_model,filepath = "code/my_first_pretrained_model.h5")




#### Testing #####

test <- readJPEG(files$mask[1])
mean(test)
mean(readJPEG(files$mask[1]))

for (i in 1:nrow(files)) {
  
  ifelse(mean(readJPEG(files$mask[i])) < 0.5, 
         file.copy(files$img[i], "C:/Users/esanc/Desktop/ESA/3_Study/1_MSc_LOEK/M8_Fernerkundung/Projektseminar/data/training/f"),
         file.copy(files$img[i], "C:/Users/esanc/Desktop/ESA/3_Study/1_MSc_LOEK/M8_Fernerkundung/Projektseminar/data/training/t"))

  }


# pb = txtProgressBar(min = 0, max = length(ind), initial = 0)
pb = txtProgressBar(min = 0, max = 4000, initial = 0)

for (i in 1:4000) {
  
  ifelse(mean(readJPEG(files$mask[i])) < 0.5, 
         file.copy(files$img[i], "C:/Users/esanc/Desktop/test/f"),
         file.copy(files$img[i], "C:/Users/esanc/Desktop/test/t"))
  
  setTxtProgressBar(pb,i)
  close(pb)
}

