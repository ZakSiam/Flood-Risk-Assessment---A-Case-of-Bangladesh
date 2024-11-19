library(ROSE)

# Importing Train Datasets

training_set1<-read.csv(file.choose(),header=TRUE) #read train dataset
test_set1<-read.csv(file.choose(),header=TRUE) #read test dataset

# Feature Matrix and Target Data

x1train = as.matrix(training_set1[,-12])
y1train = as.matrix(training_set1[,12])

x1test = as.matrix(test_set1[,-12])
y1test = as.matrix(test_set1[, 12])

library(keras)

# Initializing the Weights
init_w = initializer_random_normal(mean = 0, stddev = 0.05, seed = 123)
init_B = initializer_zeros()

########################################

# Creating the Model: "Dropout ADAM-ReLU-Sigmoid-DNN" 

model1 <- keras_model_sequential()

model1 %>% 
  layer_dense(name = "DeepLayer1",
              units = 8,
              activation = "relu",
              kernel_initializer = init_w,
              input_shape = c(11)) %>% 
  layer_dropout(0.6) %>%
  layer_dense(name = "DeepLayer2",
              units = 8,
              activation = "relu") %>% 
  layer_dropout(0.6) %>%
  layer_dense(name = "DeepLayer3",
              units = 8,
              activation = "relu") %>%
  layer_dropout(0.6) %>%
  layer_dense(name = "OutputLayer",
              units = 1,
              activation = "sigmoid")

summary(model1)

# Compiling the model

model1 %>% compile(loss = "binary_crossentropy",
                   optimizer = "adam",
                   metrics = c("accuracy"))

# Fitting Train Data Split 1

history1 <- model1 %>% 
  fit(x1train,
      y1train,
      epoch = 50,
      batch_size = 32,
      validation_split = 0.1,
      verbose = 2)
plot(history1)

# Model Evaluation Test 1: OA and Overall Loss

model1 %>% 
  evaluate(x1test,
           y1test)

# Class Prediction for Test 1

pred1 <- model1 %>% 
  predict_classes(x1test)
write.table(pred1,file="results/Predicted_Classes_Dropout_Sigmoid.csv",sep=",")

# Confusion Matrix for Test 1

table(Predicted = pred1,
      Actual = y1test)

# Output Result for Test 1

prob1 <- model1 %>% 
  predict_proba(x1test)
write.table(prob1,file="results/Predicted_Scores_Dropout_Sigmoid.csv",sep=",")

library(hydroGOF)
library(caret)

# MSE for Test 1

RMSE1=RMSE(prob1,y1test) #Calculate RMSE
MSE1 = RMSE1^2
MSE1
