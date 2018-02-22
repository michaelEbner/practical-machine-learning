library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(ggplot2)
setwd("/Users/mickey/Documents/GitHub/practical-machine-learning")


if (!file.exists("train.csv")){
  fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(fileURL, "train.csv", method="curl")
}  
if (!file.exists("test.csv")) { 
  fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileURL, "test.csv", method="curl")
}

train <- read.csv("train.csv",na.strings=c("NA","DIV/0!",""))
test <- read.csv("test.csv",na.strings=c("NA","DIV/0!",""))


train<- train[,colSums(is.na(train)) == 0] %>% select(-c(1:7))
test <- test[,colSums(is.na(test)) == 0] %>% select(-c(1:7))

set.seed(24)

sub <- createDataPartition(y=train$classe,p=0.75,list=F)
sub_train <- train[sub,]
sub_test <- train[-sub,]

mod1 <- rpart(classe~.,data=sub_train,method="class")
pred1 <- predict(mod1,data=sub_test,type="class")
rpart.plot(mod1,main="Classification Tree", extra=102,under=T,faclen=0)
confusionMatrix(pred1, sub_test$classe)

mod2 <- randomForest(classe~.,data=sub_train,method="class")
pred2 <- predict(mod2,data=sub_test,type="class")
confusionMatrix(pred2,sub_test$classe)

predictfinal <- predict(mod2, test, type="class")
predictfinal