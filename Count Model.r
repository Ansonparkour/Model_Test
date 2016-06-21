###################################################
##Task: Compare the running time of different modelS 
##Model: COUNT MODELS 
##Model Options: All Default
##Data: 6/10/2016 
##Author: Sean Zhang
####################################################

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
#count data, with 90% zero value
test_data$X5 <- sample(c(0,1),100000,  replace = T, prob = c(0.9,0.1))

#COUNT MODELS 

#Model 8: Posisson Regression
poisson_glm <- function(){
    system.time(
        model_111 <- glm(X5 ~ X6 +X7,  data = test_data, family = poisson() )
    )
}


#Model 9: Negative Binomial Regression
library(MASS)

negbinomial <- function(){
    system.time(
        model <-  glm.nb(X5 ~ X6 +X7, data = test_data)
    )
}


#Model 10:Zero-inflated Poisson Regression
#response SHOULD have zero count
library(pscl)

zeroinf_poi <- function(){
    system.time(
        model_10 <- zeroinfl(X5 ~ X6 +X7, data = test_data, dist = 'poisson')
    )
}

#Model 11: Zero-inflated NB Regression
library(pscl)

zeroinf_nb <- function(){
    system.time(
        model_11 <- zeroinfl(X5 ~ X6 +X7, data = test_data, dist = 'negbin' )
    )
}

#Model 12: Zero-truncated Poisson
#response cannot have zero count
library(VGAM)

zerotrunc_poi <- function(){
    system.time(
        model_12 <- vglm(X5 ~ X6 +X7, data = test_data, family = pospoisson())
    )
}

#Model 13: Zero-truncated NB
#response cannot have zero count
library(VGAM)

zerotrunc_nb <- function(){
    system.time(
        model_13 <- vglm(X5 ~ X6 +X7, data = test_data, family = posnegbinomial())
    )
}


t_poisson_glm <- poisson_glm()
t_negbinomial <- negbinomial()
t_zeroinf_poi <- zeroinf_poi()
t_zeroinf_nb <- zeroinf_nb()
t_zerotrunc_poi <- zerotrunc_poi()
t_zerotrunc_nb <- zerotrunc_nb()


exe_time <- rbind(t_poisson_glm,t_negbinomial,t_zeroinf_poi,t_zeroinf_nb,t_zerotrunc_poi,t_zerotrunc_nb)
exe_time <- as.data.frame(exe_time) 

#plot the excution time
par(las=2) # make label text perpendicular to axis
par(mar=c(5,8,4,3)) # increase y-axis margin.
p <- barplot(
    (exe_time$elapsed),
    horiz=TRUE,
    xlab = "Execution total time",
    names.arg = c("poisson_glm", "negbinomial", "zeroinf_poi ","zeroinf_nb ","zerotrunc_poi","zerotrunc_nb"),
    cex.names=0.7
)
text(x = exe_time$elapsed + 1.5, y = p, labels = round(exe_time$elapsed, 2),xpd = T) 






