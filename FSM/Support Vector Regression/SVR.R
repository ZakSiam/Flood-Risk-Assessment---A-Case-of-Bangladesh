library(metaheuristicOpt)
library(e1071)
library(caTools)
library(kernlab)
library(ROSE)
library(Metrics)
library(GA)

training_set1<-read.csv(file.choose(),header=TRUE) #read train dataset
test_set1<-read.csv(file.choose(),header=TRUE) #read test dataset

# Number of Support Vectors
modelSVM <- ksvm( 
  Flood_Inventory ~ ., 
  data = training_set1, 
  type = "eps-svr",
  kernel = "vanilladot",
  cross = 10
)
modelSVM@nSV
error <- mse(test_set1$Flood_Inventory, predict( modelSVM, test_set1[,-12] ) )
error
prediction <- predict( modelSVM, test_set1[,-12] )

write.table(prediction,file="results/Predicted_SVR.csv",sep=",")
