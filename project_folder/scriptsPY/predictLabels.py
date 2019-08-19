###############################################################################
###############################################################################
####### The following code was written by:
####### RK 
####### Erica Moreira
####### Maja Garbulinska
#######
###############################################################################
###############################################################################

# predict labels on the training set and save to the temp label path 
# predict only for pictures with indeces for training
predict(model, X_train, ind=indeces, labelssave=labels_path)
        