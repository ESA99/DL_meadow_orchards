# Setup on the Linux Super-Computer

#### Packages ####
library(terra)
library(caret)
library(CAST)
library(sf)
library(stars)
library(rsample)
library(tidyverse)
library(tfdatasets)
library(ggplot2)
library(mapview)
library(tfdatasets)
library(rsample)
library(gtools)

library(purrr)
library(magick)
library(jpeg)
library(lubridate)

library(neuralnet)
library(stringr)
library(pbapply)
library(ggplot2)
library(ggplotify)
library(gridExtra)

#### Setup ####
library(reticulate)

# reticulate::use_python_version("3.11")

library(tensorflow)
library(keras3)
keras3::use_backend("tensorflow")

reticulate::py_config()
tensorflow::tf_config()


num_cores <- 24  # Change this to the number of cores you want to use

tf$config$threading$set_intra_op_parallelism_threads(num_cores)
tf$config$threading$set_inter_op_parallelism_threads(num_cores)



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
  v <- vrt::vrt(ff, "dem.vrt")
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
# Crop tiles/mask after QGis

t <- list.files("/home/studies/ESA/data/auswahl/tiles/", pattern = ".tif$", full.names = T)
m <- list.files("/home/studies/ESA/data/auswahl/tiles/masks/", pattern = ".tif$", full.names = T)


### Function to create subsets with continuous naming:
dl_subsets <- function(inputrst, targetsize, targetdir, start_index = 1, is_mask = FALSE) {
  targetsizeX <- targetsize[1]
  targetsizeY <- targetsize[2]
  inputX <- ncol(inputrst)
  inputY <- nrow(inputrst)
  
  # Adjust dimensions to be divisible by targetsize
  while(inputX %% targetsizeX != 0) {
    inputX <- inputX - 1  
  }
  while(inputY %% targetsizeY != 0) {
    inputY <- inputY - 1    
  }
  
  # Determine difference and crop raster
  diffX <- ncol(inputrst) - inputX
  diffY <- nrow(inputrst) - inputY
  newXmin <- floor(diffX / 2)
  newXmax <- ncol(inputrst) - ceiling(diffX / 2) - 1
  newYmin <- floor(diffY / 2)
  newYmax <- nrow(inputrst) - ceiling(diffY / 2) - 1
  rst_cropped <- suppressMessages(crop(inputrst, raster::extent(inputrst, newYmin, newYmax, newXmin, newXmax)))
  
  agg <- suppressMessages(aggregate(rst_cropped[[1]], c(targetsizeX, targetsizeY)))
  agg[] <- suppressMessages(1:ncell(agg))
  agg_poly <- suppressMessages(rasterToPolygons(agg))
  names(agg_poly) <- "polis"
  
  total_subsets <- ncell(agg)  # Total number of subsets
  pb <- txtProgressBar(min = 0, max = total_subsets, style = 3)
  
  for(i in 1:total_subsets) {
    setTxtProgressBar(pb, i)
    e1 <- raster::extent(agg_poly[agg_poly$polis == i,])
    
    subs <- suppressMessages(crop(rst_cropped, e1))
    # Rescale to 0-1, for jpeg export
    if(!is_mask){
      subs <- suppressMessages((subs - cellStats(subs, "min")) / (cellStats(subs, "max") - cellStats(subs, "min")))
    }
    
    # Write JPEG
    target_filename <- paste0(targetdir, start_index, ".jpg")
    writeJPEG(as.array(subs), target = target_filename, quality = 1)
    start_index <- start_index + 1
  }
  
  close(pb)
  rm(subs, agg, agg_poly)
  gc()
  
  return(start_index) # Return the next index to continue from
}

dl_subsets_tif <- function(inputrst, targetsize, targetdir, start_index = 1, is_mask = FALSE) {
  targetsizeX <- targetsize[1]
  targetsizeY <- targetsize[2]
  inputX <- ncol(inputrst)
  inputY <- nrow(inputrst)
  
  # Adjust dimensions to be divisible by targetsize
  while(inputX %% targetsizeX != 0) {
    inputX <- inputX - 1  
  }
  while(inputY %% targetsizeY != 0) {
    inputY <- inputY - 1    
  }
  
  # Determine difference and crop raster
  diffX <- ncol(inputrst) - inputX
  diffY <- nrow(inputrst) - inputY
  newXmin <- floor(diffX / 2)
  newXmax <- ncol(inputrst) - ceiling(diffX / 2) - 1
  newYmin <- floor(diffY / 2)
  newYmax <- nrow(inputrst) - ceiling(diffY / 2) - 1
  rst_cropped <- suppressMessages(crop(inputrst, raster::extent(inputrst, newYmin, newYmax, newXmin, newXmax)))
  
  agg <- suppressMessages(aggregate(rst_cropped[[1]], c(targetsizeX, targetsizeY)))
  agg[] <- suppressMessages(1:ncell(agg))
  agg_poly <- suppressMessages(rasterToPolygons(agg))
  names(agg_poly) <- "polis"
  
  total_subsets <- ncell(agg)  # Total number of subsets
  pb <- txtProgressBar(min = 0, max = total_subsets, style = 3)
  
  for(i in 1:total_subsets) {
    setTxtProgressBar(pb, i)
    e1 <- raster::extent(agg_poly[agg_poly$polis == i,])
    
    subs <- suppressMessages(crop(rst_cropped, e1))
    # Rescale to 0-1, for jpeg export
    if(!is_mask){
      subs <- suppressMessages((subs - cellStats(subs, "min")) / (cellStats(subs, "max") - cellStats(subs, "min")))
    }
    
    # Write JPEG
    target_filename <- paste0(targetdir,"/", start_index, ".tif")
    writeRaster(subs[[1]],target_filename)
    start_index <- start_index + 1
  }
  
  close(pb)
  rm(subs, agg, agg_poly)
  gc()
  
  return(start_index) # Return the next index to continue from
}


# Initialize the counter and tracking data frame outside the loop
counter <- 1
image_tracking <- data.frame(InputImage = character(), StartIndex = integer(), EndIndex = integer(), stringsAsFactors = FALSE)

# Main loop including a tracking function
for (j in 1:length(t)) {
  size <- c(448, 448)
  
  dir_img1 <- "data/auswahl/temp/"
  img1 <- raster::stack(t[j])
  
  # Process image and get the new counter
  counter <- dl_subsets_tif(img1, size, dir_img1, start_index = counter)
  
  # Calculate the end index based on the number of subsets
  total_subsets <- ncell(aggregate(raster::crop(img1, raster::extent(img1)), c(size[1], size[2])))
  end_index_img1 <- counter - 1
  
  image_tracking <- rbind(image_tracking, data.frame(InputImage = basename(t[j]), StartIndex = counter - total_subsets, EndIndex = end_index_img1))
}

write.table(image_tracking, "data/auswahl/image_legend.txt", sep = " ")
subsets_legend <- image_tracking
print(subsets_legend)


# Same for the masks

counter <- 1
image_tracking <- data.frame(InputImage = character(), StartIndex = integer(), EndIndex = integer(), stringsAsFactors = FALSE)

for (j in 1:length(m)) {
  size <- c(448, 448)
  
  dir_mask1 <- "data/auswahl/masks/"
  mask1 <- raster::stack(m[j])
  
  # Process image and get the new counter
  counter <- dl_subsets(mask1, size, dir_mask1, start_index = counter, is_mask = TRUE)
  
  # Calculate the end index based on the number of subsets
  total_subsets <- ncell(aggregate(raster::crop(mask1, raster::extent(mask1)), c(size[1], size[2])))
  end_index_mask1 <- counter - 1
  
  image_tracking <- rbind(image_tracking, data.frame(InputImage = basename(m[j]), StartIndex = counter - total_subsets, EndIndex = end_index_mask1))
}

masks_legend <- image_tracking
print(masks_legend)




table_grob <- tableGrob(subsets_legend)
(table_plot <- ggplotify::as.ggplot(table_grob))

ggsave("data/auswahl/subsets_legend.png", plot = table_plot, width = 10, height = 6, dpi = 300)






#### Augmentation ####

files <- data.frame( #get paths
  img = list.files("data/auswahl/subsets/", full.names = TRUE, pattern = "*.jpg"),
  mask = list.files("data/auswahl/masks/", full.names = TRUE, pattern = "*.jpg"))

# split the data into training and validation datasets. 
files <- initial_split(files, prop = 0.8)

# prepare data for training
training_dataset <- dl_prepare_data(training(files),train = TRUE,model_input_shape = c(448,448),batch_size = 10L)
validation_dataset <- dl_prepare_data(testing(files),train = FALSE,model_input_shape = c(448,448),batch_size = 10L)

# get all tensors through the python iterator
training_tensors <- training_dataset%>%as_iterator()%>%iterate()

#how many tensors?
length(training_tensors)






#### pretrained unet_model####

vgg16_feat_extr <- application_vgg16(weights = "imagenet", include_top = FALSE, input_shape = c (448,448,3))

# optionally freeze first layers to prevent changing of their weights, either whole convbase or only certain layers
# freeze_weights(vgg16_feat_extr) #or:
# freeze_weights(vgg16_feat_extr, to = "block1_pool") 

# we'll not use the whole model but only up to layer 15
unet_tensor <- vgg16_feat_extr$layers[[15]]$output 

## add the second part of 'U' for segemntation ##

# "bottom curve" of U-net
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1024, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1024, kernel_size = 3, padding = "same", activation = "relu")

# upsampling block 1
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 512, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[14]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 512, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 512, kernel_size = 3, padding = "same", activation = "relu")

# upsampling block 2
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 256, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[10]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256, kernel_size = 3, padding = "same", activation = "relu")

# upsampling block 3
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 128, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[6]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = 3, padding = "same", activation = "relu")

# upsampling block 4
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 64, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[3]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = 3, padding = "same", activation = "relu")

# final output 
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1, kernel_size = 1, activation = "sigmoid")

# create model from tensors
pretrained_unet <- keras_model(inputs = vgg16_feat_extr$input, outputs = unet_tensor)



#### Predict ####

sample <- floor(runif(n = 1,min = 1,max = 4))
img_path <- as.character(testing(files)[[sample,1]])
mask_path <- as.character(testing(files)[[sample,2]])
img <- magick::image_read(img_path)
mask <- magick::image_read(mask_path)
pred <- magick::image_read(as.raster(predict(object = pretrained_unet,validation_dataset)[sample,,,]))

out <- magick::image_append(c(
  magick::image_append(mask, stack = TRUE),
  magick::image_append(img, stack = TRUE), 
  magick::image_append(pred, stack = TRUE)
)
)

plot(out)



# Test data set, for the prediction
test_dataset <- dl_prepare_data(train = F,predict = T,subsets_path="./data/auswahl/subsets/",model_input_shape = c(448,448),batch_size = 5L)

system.time(predictions <- predict(pretrained_unet,test_dataset))
# saveRDS(predictions, file = paste0("data/auswahl/pred_unet_",today(),".rds"))





#### Inspection ####

#visualize layers 3 and 10, channels 1 to 20
par(mfrow=c(4,5),mar=c(1,1,1,1),cex=0.5)
plot_layer_activations(img_path = "./data/auswahl/subsets/803.jpg", model=pretrained_unet ,activations_layers = c(3,10), channels = 1:20)
par(mfrow=c(1,1))




### Prediction before####
predictions <- predict(pretrained_model, dataset)
predictions <- array(data= rep(predictions,448*448),dim = c(length(predictions),448,448,1))

# Save to RDS file
saveRDS(predictions, file = paste0("data/auswahl/predictions",today(),".rds"))

# predictions <- readRDS("data/auswahl/predictions2024-07-23")



#### Create maps ####

# 1. Fix the image tracking data frame:
# for (r in 1:nrow(image_tracking)) {
#     image_tracking$StartIndex[r+1] <- image_tracking$EndIndex[r-1]+1
# }

# 2.Write the info of the prediction results into the tifs created before to give them spatial infos

image_tracking <- read.table("data/auswahl/image_legend.txt", sep = " ")
# predictions <- readRDS("data/auswahl/predictions2024-07-23.rds")

f <- list.files("/home/studies/ESA/data/auswahl/tifs/", full.names = T, pattern = ".tif") # tifs to overwrite with information
mixedsort(f)
f <- mixedsort(sort(f))

targetdir <- "data/auswahl/map_test/"

for (i in 1:nrow(image_tracking)) {
  
  a <- image_tracking$StartIndex[i] # from this file
  o <- image_tracking$EndIndex[i] # to this file
  
  # tiflist <- list()
  for (u in a:o) {
  
    rastertif <- rast(f[u])
    # rastertif[[1]][] <- unique(c(predictions[u,,,]))
    rastertif[[1]][] <- c(predictions[u,,,])
  
    # tiflist[[length(tiflist)+1]] <- rastertif
    terra::writeRaster(rastertif, paste0(targetdir,i,"_",u,".tif") )
  }
  
  # finalmap <- do.call(merge,tiflist)
  # writeRaster(finalmap, paste0(targetdir,"result_",image_tracking$InputImage[i]))
}






#### New maps attempt ####


for (i in 1:nrow(image_tracking)) {
  
  a <- image_tracking$StartIndex[i] # from this file
  o <- image_tracking$EndIndex[i] # to this file
  
    
    rebuild_img(predictions[a:o,,,,drop = FALSE], out_path = "data/auswahl/pred_out/", target_rst = stack(paste0("data/auswahl/tiles/",image_tracking$InputImage[i])) )
  
}

test <- predictions[a:o,,,]
test[]

pred_subsets <- predictions[1:20,,,,drop = FALSE]
out_path <- "/home/studies/ESA/data/auswahl/pred_out/"
target_rst <- stack(paste0("data/auswahl/tiles/",image_tracking$InputImage[1]))

rebuild_img <- function(pred_subsets,out_path,target_rst){
  require(raster)
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
  v <- vrt::vrt(ff, "dem.vrt")
  writeRaster(v, "out.tif")
}



rebuild_img <- function(pred_subsets, out_path, target_rst) {
  library(raster)
  library(stars)
  library(gtools)  # for mixedsort
  
  subset_pixels_x <- ncol(pred_subsets[1,,,])
  subset_pixels_y <- nrow(pred_subsets[1,,,])
  
  # Convert the target raster to a stars object
  target_stars <- st_as_stars(target_rst, proxy = FALSE)
  
  target_width <- ncol(target_stars)
  target_height <- nrow(target_stars)
  
  # Determine the number of tiles
  tiles_rows <- target_height / subset_pixels_y
  tiles_cols <- target_width / subset_pixels_x
  
  # Check if dimensions are divisible
  if (tiles_rows %% 1 != 0 || tiles_cols %% 1 != 0) {
    stop("Dimensions of target_rst are not divisible by dimensions of pred_subsets.")
  }
  
  # Prepare the output folder
  result_folder <- file.path(out_path, "out")
  if (dir.exists(result_folder)) {
    unlink(result_folder, recursive = TRUE)
  }
  dir.create(result_folder, showWarnings = FALSE)
  
  # Loop through each tile and write predictions to TIFF files
  for (crow in 1:tiles_rows) {
    for (ccol in 1:tiles_cols) {
      i <- (crow - 1) * tiles_cols + ccol
      
      dimx <- c((ccol - 1) * subset_pixels_x + 1, ccol * subset_pixels_x)
      dimy <- c((crow - 1) * subset_pixels_y + 1, crow * subset_pixels_y)
      
      # Ensure the index is within bounds
      if (i > length(pred_subsets[,1,1,1])) {
        stop("Index out of bounds for pred_subsets array.")
      }
      
      # Extract the subset and transpose it correctly
      subset <- pred_subsets[i,,,1]
      if (is.null(dim(subset)) || length(dim(subset)) != 2) {
        stop("Predicted subset dimension is invalid.")
      }
      
      # Create a stars object for the subset
      cstars <- st_as_stars(t(subset))
      
      if (length(st_dimensions(cstars)) < 2) {
        stop("Failed to create stars object with proper dimensions.")
      }
      
      # Set dimensions using the original raster
      target_dims <- st_dimensions(target_stars[, dimx[1]:dimx[2], dimy[1]:dimy[2]])
      st_dimensions(cstars) <- target_dims[1:2]
      
      # Write the stars object to a TIFF file
      output_file <- file.path(result_folder, paste0("_out_", i, ".tif"))
      write_stars(cstars, dsn = output_file)
    }
  }
  
  # List and sort the TIFF files
  tif_files <- list.files(path = result_folder, pattern = "\\.tif$", full.names = TRUE)
  tif_files <- mixedsort(tif_files)
  
  # Combine the TIFF files into a single mosaic using GDAL command-line tools
  mosaic_file <- file.path(result_folder, "out.tif")
  
  # Create the mosaic using gdal_merge.py
  system(paste("gdal_merge.py -o", mosaic_file, paste(tif_files, collapse = " ")))
}


#  usage
# pred_subsets <- predictions[1:20,,,,drop = FALSE]  
# out_path <- "/home/studies/ESA/data/auswahl/pred_out/"
# target_rst <- stack(paste0("data/auswahl/tiles/", image_tracking$InputImage[1]))
rebuild_img(pred_subsets, out_path, target_rst)
