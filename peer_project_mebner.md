Objective
=========

The goal of your project is to predict the manner in which they did the
exercise. This is the "classe" variable in the training set. You may use
any of the other variables to predict with. You should create a report
describing how you built your model, how you used cross validation, what
you think the expected out of sample error is, and why you made the
choices you did. You will also use your prediction model to predict 20
different test cases.

Analysis
========

Analysis steps
--------------

. Load the data and required packages . Clean data . Partitioning the
data . Train models . Model comparison . Submission . Appendix

Load the data and required packages
-----------------------------------

In this step we get all the packages needed for this task. Furthermore
we load the data in case it's not yeat stored in the working directory.

    library(caret)

    ## Warning: package 'caret' was built under R version 3.4.3

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning in as.POSIXlt.POSIXct(Sys.time()): unknown timezone 'default/
    ## Australia/Hobart'

    library(randomForest)

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(rpart)
    library(rpart.plot)
    library(tidyverse)

    ## Loading tidyverse: tibble
    ## Loading tidyverse: tidyr
    ## Loading tidyverse: readr
    ## Loading tidyverse: purrr
    ## Loading tidyverse: dplyr

    ## Warning: package 'tibble' was built under R version 3.4.1

    ## Warning: package 'tidyr' was built under R version 3.4.1

    ## Warning: package 'purrr' was built under R version 3.4.1

    ## Warning: package 'dplyr' was built under R version 3.4.1

    ## Conflicts with tidy packages ----------------------------------------------

    ## combine(): dplyr, randomForest
    ## filter():  dplyr, stats
    ## lag():     dplyr, stats
    ## lift():    purrr, caret
    ## margin():  ggplot2, randomForest

    library(ggplot2)
    setwd("/Users/mebner/Documents/for_me/R_coursera/GitHub/practical-machine-learning")


    if (!file.exists("train.csv")){
      fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
      download.file(fileURL, "train.csv", method="curl")
    }  
    if (!file.exists("test.csv")) { 
      fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
      download.file(fileURL, "test.csv", method="curl")
    }

Clean data
----------

First we unify corrupted values like NAs, empty cells or DIV/0!.

Second we are removing unnecassary columns that . are mostly empty (&lt;
60%) . contain non numerica data which can't be used for predicting

    # Replace all missing fields with NA 
    train <- read.csv("train.csv",na.strings=c("NA","DIV/0!",""))
    test <- read.csv("test.csv",na.strings=c("NA","DIV/0!",""))

    # Remove first 7 columns which can't be used for predicting
    train   <-train[,-c(1:7)]
    test <-test[,-c(1:7)]

    #Remove columns with more than 60% NAs
    nzv_col <- nearZeroVar(train)
    train <- train[, -nzv_col]

    corrupted_col <- sapply(train, function(x) {sum(!(is.na(x) | x == ""))})
    null_col <- names(corrupted_col[corrupted_col < 0.6 * length(train$classe)])
    train <- train[, !names(train) %in% null_col]

Running the Models
==================

Partition the data
------------------

    subs <- createDataPartition(y=train$classe, p=0.75, list=FALSE)
    subtrain <- train[subs, ] 
    subtest <- train[-subs, ]

1st Approach: Decision Tree
---------------------------

    #Decision Tree
    mod1 <- rpart(classe ~ ., data=subtrain, method="class")
    pred1 <- predict(mod1, subtest, type = "class")
    confusionMatrix(pred1, subtest$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1287  193   14   78   55
    ##          B   54  580  110   43  100
    ##          C   24   71  654  129   72
    ##          D   13   79   54  517   99
    ##          E   17   26   23   37  575
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.7367         
    ##                  95% CI : (0.7242, 0.749)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.6651         
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9226   0.6112   0.7649   0.6430   0.6382
    ## Specificity            0.9031   0.9224   0.9269   0.9402   0.9743
    ## Pos Pred Value         0.7910   0.6539   0.6884   0.6785   0.8481
    ## Neg Pred Value         0.9670   0.9081   0.9492   0.9307   0.9229
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2624   0.1183   0.1334   0.1054   0.1173
    ## Detection Prevalence   0.3318   0.1809   0.1937   0.1554   0.1383
    ## Balanced Accuracy      0.9128   0.7668   0.8459   0.7916   0.8062

2nd Approach: Random Forrest
----------------------------

    #Random Forest
    mod2 <- randomForest(classe ~. , data=subtrain, method="class")
    pred2 <- predict(mod2, subtest)
    confusionMatrix(pred2, subtest$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    3    0    0    0
    ##          B    1  944    3    0    0
    ##          C    0    2  851    4    0
    ##          D    0    0    1  798    2
    ##          E    0    0    0    2  899
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9963          
    ##                  95% CI : (0.9942, 0.9978)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9954          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9993   0.9947   0.9953   0.9925   0.9978
    ## Specificity            0.9991   0.9990   0.9985   0.9993   0.9995
    ## Pos Pred Value         0.9979   0.9958   0.9930   0.9963   0.9978
    ## Neg Pred Value         0.9997   0.9987   0.9990   0.9985   0.9995
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2843   0.1925   0.1735   0.1627   0.1833
    ## Detection Prevalence   0.2849   0.1933   0.1748   0.1633   0.1837
    ## Balanced Accuracy      0.9992   0.9969   0.9969   0.9959   0.9986

Comparison
==========

It's save to say that the random forrest method is performing better.
It's much more accurate: 99.59% compared to 71.88%. The out of sample
error is &lt; 0.005 (1 - accuracy for predictions made against the cross
validation subset = 1 - 0.9959 = &lt; 0.005).

Submission
----------

Finally we can use the selected model (RF) to predict the outcomes of
the original test data set.

    # predict the outcome of the original testing data set
    predict_final <- predict(mod2, test, type="class")
    predict_final

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Appendix
========

Decision Tree visualisation
---------------------------

    rpart.plot(mod1, main="Classification Tree", extra=102, under=TRUE, faclen=0)

![](peer_project_mebner_files/figure-markdown_strict/Appendix:%20Decision%20Tree%20visualisation-1.png)
