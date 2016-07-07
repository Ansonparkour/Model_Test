##################
#xgboost 
##################

#load library
library(data.table)
library(caret)
library(dplyr)
library(FSelector) #as.simple.formula
library(xgboost)


#============================SETUP DATA=============================

#load sample data 
load("sample_data.RData")
#load whole data
load("sub_data.RData")

#load the variable selected by gmb
myvariable <- fread("variable_gmb.txt")
myvariable <- as.vector(myvariable$Chg_Predictions)
length(myvariable)   #96
f <- as.simple.formula(myvariable,"Paid_3m_GE_5K")


#equaly divide the sample data to two part
set.seed(2016)
n = dim(sample_data)[1]
index = sample(n, round(0.5*n))
dataPartA = sample_data[index,]     # 3258  96
dataPartB  = sample_data[-index,]   # 3258  96


#save liner target
TargetA <- dataPartA$Paid_3m_Y3
sample_data$Paid_3m_Y3 <- NULL
TargetB <- dataPartB$Paid_3m_Y3
sample_data$Paid_3m_Y3 <- NULL


#save label for two parts
labelA = dataPartA$Paid_3m_GE_5K
table(labelA)
#0    1 
#3098  160 
labelB = dataPartB$Paid_3m_GE_5K
table(labelB)
#0    1 
#3115  143 


#just keep the variable selected before
remove = setdiff(names(dataPartA), myvariable)
dataPartA[,(remove):=NULL]
remove = setdiff(names(dataPartB), myvariable)
dataPartB[,(remove):=NULL]


#=======================Setup for xgboost matrix======================
#change to matrix
dataPartA = as.matrix(dataPartA)
#xgb matrix
trainA = xgb.DMatrix(data = dataPartA, label =labelA )
dataPartB = as.matrix(dataPartB)
trainB = xgb.DMatrix(data = dataPartB, label =labelB )



#==============================xgboost model============================
#watchlist
watchlist = list(train=trainA, test=trainB)

#use dataPartA(trainA) to train the model and predict dataPartB(trainB) target
bstA = xgb.train(data = trainA,
                 mac.depth = 4, 
                 eta=0.1, 
                 nround=1000,
                 colsampleby_tree=0.75,
                 subsample=0.75,
                 min_chind_weight=10, 
                 gamma=0.1, 
                 nthread=16, 
                 objective="binary:logistic", 
                 eval.metric = "auc",
                 watchlist=watchlist,
                 early.stop.round = 20, 
                 maximize = TRUE,
                 alpha=0.01,
                 lambda=1, 
                 booster="gblinear")

#Stopping. Best iteration: 888
#train-auc:0.878750  test-auc:0.720234

#save model
xgb.save(bstA, "ModelA")

#use trainA model predict trainB data 
bstA.pred = predict(bstA, trainB)
#cutoff = 0.5 
bstA.pred.label = ifelse(bstA.pred>0.5, 1, 0)
#comapre prediction label of trianB with true label of trainB 
confusionMatrix(bstA.pred.label ,labelB)

#             Reference
##Prediction    0    1
#      0      3078  118
#      1       37   25

#Accuracy : 0.9524  
#Sensitivity : 0.9881          
#Specificity : 0.1748  


#------------------------seconde part----------------------------------------

#watchlist
watchlist = list(train=trainA, test=trainB)

#use dataPartB(trainB) to train the model and predict dataPartA(trainA) target
bstB = xgb.train(data = trainB,
                 mac.depth = 4, 
                 eta=0.1, 
                 nround=1000,
                 colsampleby_tree=0.75,
                 subsample=0.75,
                 min_chind_weight=10, 
                 gamma=0.1, 
                 nthread=16, 
                 objective="binary:logistic", 
                 eval.metric = "auc",
                 watchlist=watchlist,
                 early.stop.round = 20, 
                 maximize = TRUE,
                 alpha=0.01,
                 lambda=1, 
                 booster="gblinear")


#Stopping. Best iteration: 170
#train-auc:0.718815  test-auc:0.833802

#save model
xgb.save(bstB, "ModelB")

#use trainB model predict trainA data 
bstB.pred = predict(bstB, trainA)
#cutoff = 0.5 
bstB.pred.label = ifelse(bstB.pred>0.5, 1, 0)
#comapre prediction label of trianB with true label of trainB 
confusionMatrix(bstB.pred.label ,labelA)

#Confusion Matrix and Statistics
#            Reference
#Prediction    0    1
#    0       3069  132
#    1        29   28

#Accuracy : 0.9506  
#Sensitivity : 0.9906          
#Specificity : 0.1750



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#====================whole data test============================#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#change sample data to sub_data test the whole data prefermence 
sample_data = sub_data


#----------------train model on trainA data--------------- 
#train = trainA data,  test = trainB 


#stopping. Best iteration: 361
#train-auc:0.840859  test-auc:0.837183

#save model
xgb.save(bstA, "xgboost_ModelA")
#bstA = xgb.load("xgboost_ModelA")


#------------------find the suitable cutoff value------------

#number of event(target = 1)
n = data.frame(table(labelA))[2,2]

#use trainA model predict trainA data 
bstA.predA = predict(bstA, trainA)
#change to two decimal places  
bstA.predA = round(bstA.predA, digits = 2)

#sort prediction as decrese order, and take n-th prediction probability as cutoff  
cutoff = sort(bstA.predA,decreasing = TRUE)[n]
#cutoff = 0.14


#----------------trainA model predict trainB data------------------------------

bstA.predB = predict(bstA, trainB)


bstA.predB.label = ifelse(bstA.predB>cutoff, 1, 0)

table(bstA.predB.label)

#comapre prediction label of trianB with true label of trainB 
confusionMatrix(bstA.predB.label ,labelB)

#Confusion Matrix and Statistics

#               Reference
#Prediction      0      1
#    0         301755  9172
#    1         8644    6221

#Accuracy : 0.9453    
#Sensitivity : 0.9722          
#Specificity : 0.4041 

#I think for our case, the sensitivity is more important than specificity, cuz our concentration 
#is the high cost of prediction = 1 



#=========================seconde part===============================================
###same way to train the model using trainB data
###use trainB data predict trainB data, compare real labelB and find suitable cuttoff
###apply trainB model to trainA data using new cutoff. 


#----------------train model on trainA data--------------- 
#train = trainA data,  test = trainB 


#Stopping. Best iteration: 387
#train-auc:0.839933	test-auc:0.837468

#save model
xgb.save(bstB, "xgboost_ModelB")
#bstB = xgb.load("xgboost_ModelB")

#------------------find the suitable cutoff value------------

#number of event(target = 1)
n = data.frame(table(labelB))[2,2]

#use trainA model predict trainA data 
bstB.predB = predict(bstB, trainB)
#change to two decimal places  
bstB.predB = round(bstB.predB, digits = 2)

#sort prediction as decrese order, and take n-th prediction probability as cutoff  
cutoff = sort(bstB.predB,decreasing = TRUE)[n]
#cutoff = 0.14

#----------------trainA model predict trainB data------------------------------

bstB.predA = predict(bstB, trainA)


bstB.predA.label = ifelse(bstB.predA>cutoff, 1, 0)


#comapre prediction label of trianB with true label of trainB 
confusionMatrix(bstB.predA.label ,labelA)

#Confusion Matrix and Statistics

#               Reference
#Prediction      0      1
#    0         301582   8741
#    1          9214    6255

#Accuracy :  0.9449    
#Sensitivity : 0.9704          
#Specificity : 0.4171 



#=============================model analysis=================================

#combine two part test result 
bstA.result = data.frame(TargetB,labelB,bstA.predB,bstA.predB.label)
colnames(bstA.result) = c("realValue", "label","prob","predictedProb" )
bstB.result = data.frame(TargetA,labelA,bstB.predA,bstB.predA.label)
colnames(bstB.result) = c("realValue", "label","prob","predictedProb" )

#using rbind should has same column name
result = rbind(bstA.result,bstB.result)

result = data.table(result)
result = result[order(-rank(result$realValue)),]

save(result,file = "xgboost_result.RData")

