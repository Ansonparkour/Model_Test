#########################
#glmnet model 
#########################

set.seed(2016)

#load library
library(data.table)
library(caret)
library(dplyr)
library(FSelector) #as.simple.formula
library(glmnet)

#============================SETUP DATA=============================

#selected variables
f.myvariable <- function(file,targetLabel){
      myvariable <- read.table(file)
      myvariable <- as.vector(myvariable$V1)
      simple.formula <- as.simple.formula(myvariable,targetLabel)
      return(list(myvariable, simple.formula))
}


#set up xgb.matrix and target label
f.glmdata <- function(sample_data,targetLabel,myvariable ){
      
      #equaly divide the sample data to two part
      set.seed(2016)
      n = dim(sample_data)[1]
      index = sample(n, round(0.5*n))
      dataPartA = sample_data[index,]     
      dataPartB = sample_data[-index,]  
      
      #save liner target
      TargetA <- dataPartA$Paid_3m_Y3
      TargetB <- dataPartB$Paid_3m_Y3
      sample_data$Paid_3m_Y3 <- NULL
      
      #remove previous target label 
      sample_data$Paid_3m_GE_5K <- NULL
      
      
      #just keep the variable selected before
      remove = setdiff(names(dataPartA), myvariable)
      dataPartA[,(remove):=NULL]
      remove = setdiff(names(dataPartB), myvariable)
      dataPartB[,(remove):=NULL]
      
      #Setup glmnet data, matrix data, x and y should be sepreated 
      trainA = as.matrix(dataPartA)
      TargetA = as.matrix(TargetA)
      trainB = as.matrix(dataPartB)
      TargetA = as.matrix(TargetA)
      
      
      returnvalue = list(trainA,trainB, TargetA, TargetB  )
      names(returnvalue) = c("trainA", "trainB",
                             "TargetA", "TargetB")
      return(returnvalue)
}


#===================grid search for alpha===============================
#lasso: alpha=1
#ridge: alpha=0
#elnet: alpha=.5


best.alpha <- function(trainx, trainy, testx, testy, lambdaType, familyname, num){
      for (i in 0:num) {
            assign(paste("fit", i, sep=""), cv.glmnet(trainx, trainy, type.measure="mse", 
                                                      alpha=i/num,family=familyname))
      }
      y.pred = NULL
      for (j in 0:num){
            y.pred <- cbind(y.pred, predict(get(paste("fit", j, sep="")), s = lambdaType, testx)) 
      }
      mse = 1:ncol(y.pred)
      for (k in 1:(num+1)){
            mse[k] <- mean((testy - y.pred[,k] )^2)
      }
      
      alpha = (which.min(mse) - 1)/num
      
      returnvalue = (list(min(mse), alpha))
      names(returnvalue) = c("mse", "alpha")
      return(returnvalue)
}

#=================glmnet model==============================================



glt <- function(trainx, trainy, testx, testy, best.alpha, trunc, modelName){
      
      trainy <- sapply(trainy , function(x) if (x >= trunc) x = trunc else x = x)
      testy <- sapply(testy , function(x) if (x >= trunc) x = trunc else x = x)
      

      set.seed(2016)
      gltA <- cv.glmnet(trainx,
                        trainy,
                        alpha= best.alpha,
                        family='gaussian')
      
      #save model, model name should be RData
      save(gltA, file= modelName)
      
      #train model predict test data
      gltA.predB = predict(gltA, testx, s = "lambda.min")
      
      returnvalue = list(gltA, gltA.predB)
      names(returnvalue) = c("model", "prediction")
      return(returnvalue)
}



#=============================save result=================================


f.result <- function(result.file.name, TargetB,model1.pred,TargetA,model2.pred){
      
      
      #combine two part test result 
      model1.result = data.frame(TargetB,model1.pred)
      colnames(model1.result) = c("realValue","predictedValue")
      
      model2.result = data.frame(TargetA,model2.pred)
      colnames(model2.result) = c("realValue","predictedValue")
      
      #using rbind should has same column name
      result = rbind(model1.result,model2.result)
      result = data.table(result)
      result = result[order(-rank(result$realValue)),]
      
      save(result,file = result.file.name)
      return(result)

}
      
#-------------------------------------------------------------------------------------
#find the best alpha and lambda
a1 = best.alpha(train,train_target, test,test_target, 0 , "gaussian" , 10)
#$mse
#[1] 25449292
#$alpha
#[1] 0.1
a2 = best.alpha(train,train_target, test,test_target, "lambda.min", "gaussian" , 10)
#$mse
#[1] 25413764
#$alpha
#[1] 0.5
a3 = best.alpha(train,train_target, test,test_target, "lambda.1se", "gaussian" , 10)
#$mse
#[1] 26889187
#$alpha
#[1] 0.5

#therefore the best aplha is from a2 with smallest mse 
best.alpha = a2$alpha #which is elnet



#============================main=======================================

#load sample data 
load("sample_data.RData")
#load whole data
load("sub_data.RData")


#change target label
targetLabel = "Paid_3m_GE_5K"
modelNmae1 = "glmnet_5k_model1"
modelNmae2 = "glmnet_5k_model2"
resultName = "Result_glmnet_5k_model.RData"


#load the variable selected
file = "variable_gmb.txt"
myvariable = f.myvariable(file,targetLabel)[[1]]

#assign the train and test data
glmData = f.glmdata(sub_data,targetLabel,myvariable )
train = glmData$trainA
test = glmData$trainB
train_target = glmData$TargetA
test_target = glmData$TargetB


#----------------------------------------------------------------------------------
trunc = 25000

model1 = glt(train,train_target, test,test_target, best.alpha, trunc,  "glt_25k_model1.RData")
model2 = glt(test,test_target, train,train_target, best.alpha, trunc, "glt_25k_model2.RData")


TargetB =sapply(test_target , function(x) if (x > trunc) x = trunc else x = x)
model1.pred = model1$prediction

TargetA = sapply(train_target , function(x) if (x > trunc) x = trunc else x = x)
model2.pred = model2$prediction

result = f.result("result_glt_25k.RData", TargetB,model1.pred,TargetA,model2.pred)













