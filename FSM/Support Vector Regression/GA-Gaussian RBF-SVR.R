library(metaheuristicOpt)
library(e1071)
library(caTools)
library(kernlab)
library(ROSE)
library(Metrics)
library(GA)

training_set1<-read.csv(file.choose(),header=TRUE) #read train dataset
test_set1<-read.csv(file.choose(),header=TRUE) #read test dataset

MsE1 <- function( training_set1, test_set1, epsilon, c, gamma ) 
{
  ## train SVM model 
  model1 <- ksvm( 
    Flood_Inventory ~ ., 
    data = training_set1, 
    type = "eps-svr",
    kernel = "rbfdot",
    epsilon = epsilon,
    C = c, 
    kpar=list(sigma=gamma),
    cross = 10
  )
  
  ## test and calculate RMSD
  MSE1 <- mse(test_set1$Flood_Inventory, predict( model1, test_set1[,-12] ) )
  
  ## return calculated RMSD
  return ( MSE1 )
}

fitness_func1 <- function( x ) 
{
  
  ## fetch SVM parameters
  epsilon_val <- x[ 1 ]
  gamma_val <- x[ 2 ]
  c_val <- x[ 3 ]
  msd_vals <- MsE1( training_set1, test_set1, epsilon_val, c_val, gamma_val ) 
  return ( -msd_vals )
}


para_value_min <- c( epsilon = 0, gamma = 1e-3, c = 1e-4 )
para_value_max <- c( epsilon = 1, gamma = 2, c = 10 )


## run genetic algorithm
results1 <- ga( type = "real-valued", 
                fitness = fitness_func1, 
                names = names( para_value_min ), 
                lower = para_value_min, 
                upper = para_value_max,
                popSize = 50, 
                maxiter = 100
)
summary(results1)
plot(results1)


# Number of Support Vectors
modelSVM <- ksvm( 
  Flood_Inventory ~ ., 
  data = training_set1, 
  type = "eps-svr",
  kernel = "rbfdot",
  epsilon = 0.1736948,
  C = 1.411367, 
  kpar=list(sigma=0.1148712),
  cross = 10
)
modelSVM@nSV
error <- mse(test_set1$Flood_Inventory, predict( modelSVM, test_set1[,-12] ) )
error
prediction <- predict( modelSVM, test_set1[,-12] )

write.table(prediction,file="results/Predicted_GA_Gaussian_SVR.csv",sep=",")
