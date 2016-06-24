###################################################
##Task: Compare the running time of different modelS 
##Model: LOGISTIC MODEL
##Model Options: All Default
##Data: 6/8/2016 
##Author: Sean Zhang
####################################################


#for some model, each execution may result slight different running time
#therefor better get mean of execution time
##execution_time = mean(replicate(5, system.time(modelFun))[3,])



#create a fake random 1000 * 5 matrix corresponding to 50 features with 1000 observations
set.seed(2016)
test_data <-  data.frame(matrix( rnorm(100000*50,mean=0,sd=1), 100000, 50)) 

#intercept 
test_data$X1 <- sample(1,100000, replace = T)
#binary 
test_data$X2 <- sample(0:1,100000, replace = T)
#ordinal
test_data$X3 <- sample(1:3,100000, replace = T)
#4 level nominal
test_data$X4 <- sample(1:4,100000, replace = T)


########################################
#Logistic Models

#Model 1 : Generalized Linear Model
#Family : binomial

binomial_glm <- function(){
    system.time(
        model <- glm(X2 ~ ., data = test_data, family = binomial())
    )
}

#Model 2 : Multinomial Logistic Regression
library(nnet)

multinomial <- function(){
    test_data$X3 <- as.factor(test_data$X3)
    #choose the level of outcome as baseline
    test_data$X3 <- relevel(test_data$X3, ref = '3')
    system.time(
        model <- multinom(X3 ~ ., data = test_data)
    )
}

#Model 3 : Ordinal Logistic Regression
#ordered logistic regression model
library(MASS)
orderedLogit <- function(){
    test_data$X3 <- as.factor(test_data$X3)
    system.time(
        model_3 <- polr(X3 ~ . , data = test_data, Hess=TRUE)
    )
}

#Model 4 : Boosted Generalized Linear Model
library(mboost)

boosted_glm <- function(){
    test_data$X3 <- as.numeric(test_data$X3)
    #each time run the model may have slightly different execution time
    system.time(
        model_4 <-  glmboost(X3 ~ ., data = test_data)
    )
}

#Model 5 : Generalized Linear Model with lasso or elasticnet
library(glmnet)

lasso_glm <- function(){
    #glmnet need a matrix of predictors, not a data frame
    x = as.matrix(data.frame(test_data))
    #cuz here choose binomial family, so y should binary 
    system.time(
        model_5 <- glmnet(x,y=test_data$X2,alpha=1,family='binomial')
    )
}


#Model 6 : Generalized Linear Model with Stepwise Feature Selection
library(MASS)
library(caret)
#model_6 <- train(X3 ~ ., data = test_data, method = "glmStepAIC")

#Model 7 : Generalized Partial Least Squares
#Error: cannot allocate vector of size 74.5 Gb
library(MASS)
library(gpls)
#model_7 <- gpls(X2 ~ X1, data = test_data, family='binomial' )


##get the execution time of each model 
t_binomial_glm <- binomial_glm()
t_multinomial <- multinomial()
t_orderedLogit <- orderedLogit()
t_boosted_glm <- boosted_glm()
t_lasso_glm <- lasso_glm()

exe_time <- rbind(t_binomial_glm,t_multinomial,t_orderedLogit,t_boosted_glm,t_lasso_glm)
exe_time <- as.data.frame(exe_time) 

#plot the excution time
library(ggplot2)
par(las=2) # make label text perpendicular to axis
par(mar=c(5,8,4,2)) # increase y-axis margin.
p <- barplot(
        (exe_time$elapsed),
        horiz=TRUE,
        xlab = "Execution total time",
        names.arg = c("binomial_glm", "multinomial", "orderedLogit","orderedLogit","lasso_glm"),
        cex.names=0.7
)
text(x = exe_time$elapsed + 0.5, y = p, labels = round(exe_time$elapsed, 1),xpd = T) 


