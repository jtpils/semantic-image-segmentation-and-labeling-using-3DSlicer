###############################################################################
###############################################################################
####### The following code was written by:
####### RK 
####### Erica Moreira
####### Maja Garbulinska
#######
###############################################################################
###############################################################################


## define transformations of the data for training and validation
## do not random flip the validation data 
transformationsTrain = Transformations()
transformationsTrain.add(RandomCropTrain((256,256)))
transformationsTrain.add(RandomFlipLR(0.5))
transformationsValid = Transformations()
transformationsValid.add(CropValid((256,256)))
# transformations.add(my_transforms.RandomBrightness(prob=0.5, limit=(-0.15, 0.15)))

## define the number of classes 
n_classes = 3

## transform the training set and put it in the data loader.  
training_set = FromImageFilenames(X_train, y_train, n_classes, transformationsTrain)
train_loader = DataLoader(training_set, batch_size=2, shuffle=True)

## transform the test set and put it in the data loader.  
test_set = FromImageFilenames(X_valid, y_valid, n_classes, transformationsValid)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

model = StackedUnet()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


if os.path.isfile(model_path):
## load in saved weights if the model was run before. 
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
    optimizer.load_state_dict(torch.load(os.path.join(model_path, "optim.pth")))

is_cuda_available = torch.cuda.is_available()

## weights 
wtperclass = torch.from_numpy(np.asarray([25.0,1.0,20.0],dtype=np.float32))
