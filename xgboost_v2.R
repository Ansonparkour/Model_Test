###################
#xgboost
###################
set.seed(2016)

#load library
library(data.table)
library(caret)
library(dplyr)
library(FSelector) #as.simple.formula
library(xgboost)

#============================SETUP DATA=============================

#selected variables
f.myvariable <- function(file,targetLabel){
      myvariable <- read.table(file)
      myvariable <- as.vector(myvariable$V1)
      simple.formula <- as.simple.formula(myvariable,targetLabel)
      return(list(myvariable, simple.formula))
}


#set up xgb.matrix and target label
f.xgbData <- function(sample_data,targetLabel,myvariable ){
      
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
      
      #save label for two parts
      labelA = subset(dataPartA, select = targetLabel)
      labelA = as.data.frame(labelA)[,1]
      labelB = subset(dataPartB, select = targetLabel)
      labelB = as.data.frame(labelB)[,1]
      
      #just keep the variable selected before
      remove = setdiff(names(dataPartA), myvariable)
      dataPartA[,(remove):=NULL]
      remove = setdiff(names(dataPartB), myvariable)
      dataPartB[,(remove):=NULL]
      
      #Setup for xgboost matrix
      dataPartA = as.matrix(dataPartA)
      trainA = xgb.DMatrix(data = dataPartA, label =labelA )
      dataPartB = as.matrix(dataPartB)
      trainB = xgb.DMatrix(data = dataPartB, label =labelB )
      
      returnvalue = list(trainA,trainB,labelA, labelB, TargetA, TargetB  )
      names(returnvalue) = c("trainA", "trainB",
                             "labelA", "labelB",
                             "TargetA", "TargetB")
      return(returnvalue)
}


#==================xgboost model==============================================



bst <- function(train, test, modelName, train_label, test_label){
      
      #watchlist
      watchlist = list(train=train, test=test)
      set.seed(2016)
      bstA =  xgb.train(data = train,
                        mac.depth = 4, 
                        eta=0.1, 
                        nround=10000,
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
      
      #save model
      xgb.save(bstA, modelName)
      
      #find the suitable cutoff value
      #number of event(target = 1)
      n = data.frame(table(train_label))[2,2]
      #use trainA model predict trainA data 
      bstA.predA = predict(bstA, train)
      #change to two decimal places  
      bstA.predA = round(bstA.predA, digits = 2)
      #sort prediction as decrese order, and take n-th prediction probability as cutoff  
      cutoff = sort(bstA.predA,decreasing = TRUE)[n]
      
      #train model predict test data
      bstA.predB = predict(bstA, test)
      bstA.predB.label = ifelse(bstA.predB>cutoff, 1, 0)
      
      #comapre prediction label of trianB with true label of trainB 
      confusion = confusionMatrix(bstA.predB.label ,test_label)
      
      
      returnvalue = (list(cutoff, confusion, bstA.predB, bstA.predB.label))
      names(returnvalue) = c("cutoff", "confusion_matrix",
                             "bstA.predB", "bstA.predB.label")
      return(returnvalue)
}


#=============================save result=================================


f.result <- function(result.file.name,
                     TargetB,labelB,model1.pred,model1.pred.label,
                     TargetA,labelA,model2.pred,model2.pred.label){
      
      #combine two part test result 
      model1.result = data.frame(TargetB,labelB,model1.pred, model1.pred.label)
      colnames(model1.result) = c("realValue", "label","prob","predictedProb")
      
      model2.result = data.frame(TargetA,labelA,model2.pred, model2.pred.label)
      colnames(model2.result) = c("realValue", "label","prob","predictedProb")
      
      #using rbind should has same column name
      result = rbind(model1.result,model2.result)
      result = data.table(result)
      result = result[order(-rank(result$realValue)),]
      
      save(result,file = result.file.name)
      return(result)
      
}



#============================main=======================================

#load sample data 
load("sample_data.RData")
#load whole data
load("sub_data.RData")


#change target label
targetLabel = "Paid_3m_GE_65K"
modelNmae1 = "xgb_65k_model1"
modelNmae2 = "xgb_65k_model2"
resultName = "Result_xgb_65k_model.RData"

#remove previous target label 
sub_data[,702] = NULL

sub_data[,Paid_3m_GE_65K:=ifelse(Paid_3m_Y3>=65000, 1, 0)]
table(sub_data$Paid_3m_GE_65K == 1)



#load the variable selected
file = "variable_gmb.txt"
myvariable = f.myvariable(file,targetLabel)[[1]]

xgbData = f.xgbData(sub_data,targetLabel,myvariable )

train = xgbData$trainA
test = xgbData$trainB
train_label = xgbData$labelA
test_label = xgbData$labelB

#---------------------------------------
model1= bst(train, test, modelNmae1, train_label, test_label)
model1$cutoff
model1$confusion_matrix


model2 = bst(test, train, modelNmae2, test_label, train_label)
model2$cutoff
model2$confusion_matrix


TargetA = xgbData$TargetA
labelA  = xgbData$labelA 
model1.pred = model1$bstA.predB
model1.pred.label = model1$bstA.predB.label
TargetB =xgbData$TargetB
labelB = xgbData$labelB 
model2.pred = model2$bstA.predB
model2.pred.label = model2$bstA.predB.label

result = f.result(resultName,
                  TargetB,labelB,model1.pred, model1.pred.label,
                  TargetA,labelA,model2.pred,model2.pred.label)

