library(ROSE)

# Importing Train and Test Datasets

training_set1<-read.csv(file.choose(),header=TRUE) #read train dataset
test_set1<-read.csv(file.choose(),header=TRUE) #read test dataset

# Feature Matrix and Target Data

x1train = as.matrix(training_set1[,-12])
y1train = as.matrix(training_set1[,12])

x1test = as.matrix(test_set1[,-12])
y1test = as.matrix(test_set1[, 12])

library(keras)

# One Hot Encoding for Softmax

y1_train <- to_categorical(y1train)
y1_test <- to_categorical(y1test)

# Initializing the Weights

init_w = initializer_random_normal(mean = 0, stddev = 0.05, seed = 123)
init_B = initializer_zeros()

########################################

# Creating the Model: "Dropout-ADAM-ReLU-Softmax-DNN"

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
              units = 2,
              activation = "softmax")

summary(model1)

# Compiling the model

model1 %>% compile(loss = "categorical_crossentropy",
                   optimizer = "adam",
                   metrics = c("accuracy"))

# Fitting Train Data Split 1

history1 <- model1 %>% 
  fit(x1train,
      y1_train,
      epoch = 50,
      batch_size = 32,
      validation_split = 0.1,
      verbose = 2)
plot(history1)

# Model Evaluation Test 1: OA and Overall Loss

model1 %>% 
  evaluate(x1test,
           y1_test)

# Class Prediction for Test 1

pred1 <- model1 %>% 
  predict_classes(x1test)
write.table(pred1,file="results/Predicted_Classes_Dropout_Softmax_DNN.csv",sep=",")

# Confusion Matrix for Test 1

table(Predicted = pred1,
      Actual = y1test)

# Output Result for Test 1

prob1 <- model1 %>% 
  predict_proba(x1test)
write.table(prob1,file="results/Predicted_Scores_Dropout_Softmax_DNN.csv",sep=",")

