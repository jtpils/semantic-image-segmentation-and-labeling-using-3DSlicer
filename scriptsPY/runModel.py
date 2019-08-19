###############################################################################
###############################################################################
####### The following code was written by:
####### RK 
####### Erica Moreira
####### Maja Garbulinska
####### 
####### Train the model. You can go to the modelFunctions.py to see how the 
####### train function is defined. 
#######
###############################################################################
###############################################################################

for epoch in range(epochs):
    train(epoch, model, optimizer, train_loader, is_cuda_available, wtperclass)
    
print("Training done")
