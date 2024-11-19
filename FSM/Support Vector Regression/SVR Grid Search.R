library(e1071) 
library(hydroGOF) 
library(caret)
library(metaheuristicOpt)
library(e1071)
library(caTools)
library(kernlab)
library(ROSE)
library(Metrics)
library(GA)

training_set1<-read.csv(file.choose(),header=TRUE) #read train dataset
test_set1<-read.csv(file.choose(),header=TRUE) #read test dataset

OptModelsvm1=tune(svm, Flood_Inventory ~ ., data=training_set1,ranges=list(gamma=seq(0.1,2,0.1),cost=seq(0.1,10,0.1)))

print(OptModelsvm1) #Print optimum value of parameters

plot(OptModelsvm1) #Plot the perfrormance of SVM Regression model

BstModel1=OptModelsvm1$best.model #Find out the best model

PredYBst1=predict(BstModel1,test_set1[,-12]) #Predict Y using best model

MSEBst1=mse(test_set1$Flood_Inventory, PredYBst1) #Calculate MSE of the best model
MSEBst1

W1 = t(BstModel1$coefs) %*% BstModel1$SV #Find value of W
b1 = BstModel1$rho #Find value of b
W1
b1

write.table(PredYBst1,file="results/Predicted_Grid_Search.csv",sep=",")
