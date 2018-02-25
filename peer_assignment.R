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


train   <-train[,-c(1:7)]
test <-test[,-c(1:7)]

nzv_col <- nearZeroVar(train)
train <- train[, -nzv_col]

corrupted_col <- sapply(train, function(x) {sum(!(is.na(x) | x == ""))})
null_col <- names(corrupted_col[corrupted_col < 0.6 * length(train$classe)])
train <- train[, !names(train) %in% null_col]

set.seed(24)

subs <- createDataPartition(y=train$classe, p=0.75, list=FALSE)
subtrain <- train[subs, ] 
subtest <- train[-subs, ]


#Decision Tree
mod1 <- rpart(classe ~ ., data=subtrain, method="class")
pred1 <- predict(mod1, subtest, type = "class")
rpart.plot(mod1, main="Classification Tree", extra=102, under=TRUE, faclen=0)
confusionMatrix(pred1, subtest$classe)

#Random Forest
mod2 <- train(classe ~ ., data=subtrain, method="rf")
pred2 <- predict(mod2, subtest)
confusionMatrix(pred2, subtest$classe)

