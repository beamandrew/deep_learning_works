library(readr)
library(h2o)
library(caret)
library(dplyr)
library(genefilter)
library(RSkittleBrewer)
library(keras)
library(genefilter)

dat = read_csv("~/Downloads/train.csv")
dat = dat %>% filter(label < 2)

library(caret)
set.seed(12345)
inTrain = createDataPartition(dat$label, p=0.8, list=FALSE)
training = dat[inTrain,]
testing = dat[-inTrain,]


local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)

ntrain = dim(training)[1]
ss = seq(10,80,by=5)
B = 5
leek = deep = matrix(NA,ncol=length(ss),nrow=B)


for(i in seq_along(ss)){
  for(b in 1:B){
    samp = createDataPartition(training$label,p=ss[i]/length(training$label))$Resample1
    training0 = training[samp,]
    tmp = colFtests(as.matrix(training0[,-1]),as.factor(training0$label))
    index = which(rank(tmp$p.value) <= 10)
    leekasso0 = glm(training0$label ~ ., data=training0[,(index + 1)],family=binomial)
    leek[b,i] = mean((predict(leekasso0,testing,type='response') > 0.5) == testing$label)
    
    model <- keras_model_sequential() 
    model %>% 
      layer_dense(units = 128, input_shape = c(784), activation = 'relu') %>%
      layer_dropout(0.5) %>% 
      layer_dense(units = 128, activation = 'relu') %>% 
      layer_dropout(0.5) %>% 
      layer_dense(units = 1, activation = 'sigmoid') %>% 
      compile(
        optimizer = optimizer_adam(lr=0.001),
        loss = 'binary_crossentropy',
        metrics = c('accuracy')
      )
    
    model %>% fit(as.matrix(training0[,-1]), as.matrix(training0[,1]), 
                  verbose=0, epochs=200, batch_size=1)
    
    score <- model %>% evaluate(as.matrix(testing[,-1]), as.matrix(testing[,1]),batch_size=128,verbose=0)
    deep[b,i] <- score[[2]]
  }
  print(i)
}

trop = RSkittleBrewer("tropical")
plot(ss,colMeans(leek),col=trop[1],type="l",
     ylim=c(0.5,1),
     xlab="Training Set Sample Size",
     ylab="Accuracy",
     main="Leekasso vs. Deep Learning Redux in R",lwd=3)
upp = apply(leek,2,quantile,0.9,na.rm=T)
low = apply(leek,2,quantile,0.1,na.rm=T)
segments(ss,low,ss,upp,col=trop[1],lwd=3)

lines(ss,colMeans(deep),col=trop[2],lwd=3)
upp = apply(deep,2,quantile,0.9,na.rm=T)
low = apply(deep,2,quantile,0.1,na.rm=T)
segments(ss,low,ss,upp,col=trop[2],lwd=3)


legend(50,0.7,
       legend=c("Top 10 (Leekasso)", "Deep Learning"),
       col=trop[1:2],lwd=3,lty=1)










