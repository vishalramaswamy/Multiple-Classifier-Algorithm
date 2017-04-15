 # ========================================
# Multiple Hypothesis Testing
# Part 1: K-fold Cross-Validation Paired t-Test
# Part 2: Analysis of Variance (ANOVA) Test
# Part 3: Wilcoxon Signed Rank test
# ========================================
install.packages("cvTools")
install.packages("C50")
install.packages("e1071")
install.packages("kernlab")
# Load the required R packages
require(cvTools)
require(C50)
require(e1071)
require(kernlab)
# **********************************************
# Part 1: K-fold Cross-Validation Paired t-Test
# *****************************************

# Load the iris data set
i_data<-read.csv("./datasets/Iris_data.txt")
nrow(i_data)
# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
set.seed(20)
i_data <- i_data[sample(nrow(i_data)),]
i_data
k <- 10 

folds <- cvFolds(NROW(i_data), K=k)
c50_error<-0
svm_error<-0


for(i in 1:k){
  train <- i_data[folds$subsets[folds$which != i], ] #Set the training set
  test <- i_data[folds$subsets[folds$which == i], ] #Set the validation set
  
  c50_decision_tree<-C5.0(x=train[,-5],y=train$Iris.setosa)
  c50_predict<-predict(c50_decision_tree,test[,-5])
  c50_error[i]<-1-(sum(c50_predict == test[,5] ) / length(c50_predict))
  
  svm_model<-svm(Iris.setosa~.,data = train)
  svm_predict<-predict(svm_model,test[,-5])
  svm_error[i]<-1-(sum(svm_predict == test[,5] ) / length(svm_predict))
}
mean(c50_error)
mean(svm_error)
test_result <- t.test(c50_error,svm_error,paired=TRUE)
test_result

#p-value 0.3434 >0.05 thus  we failto reject null hypothesis

# Use the training set to train a C5.0 decision tree and Support Vector Machine


# Make predictions on the test set and calculate the error percentages made by both the trained models


# Perform K-fold Cross-Validation Paired t-Test to compare the means of the two error percentages


# *****************************************
# Part 2: Analysis of Variance (ANOVA) Test
# *****************************************

# Load the Breast Cancer data set 
b_data<-read.csv("./datasets/Wisconsin_Breast_Cancer_data.txt")
head(b_data)
b_data<-b_data[,-1]
nrow(b_data)
# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
set.seed(20)
b_data <- b_data[sample(nrow(b_data)),]
head(b_data)
k <- 10 

folds <- cvFolds(NROW(b_data), K=k)
c50_error<-0
svm_error<-0
bayes_error<-0
logistic_error<-0

# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)
# 	3. Naive Bayes	(?naiveBayes in e1071 package) 
# 	4. Logistic Regression (?glm in stats package) 

# Make predictions on the test set and calculate the error percentages made by the trained models
for(i in 1:k){
  train <- b_data[folds$subsets[folds$which != i], ] #Set the training set
  test <- b_data[folds$subsets[folds$which == i], ] #Set the validation set
  
  c50_decision_tree<-C5.0(x=train[,-1],y=train$M)
  c50_predict<-predict(c50_decision_tree,test[,-1])
  c50_error[i]<-1-(sum(c50_predict == test[,1] ) / length(c50_predict))
  
  svm_model<-ksvm(M~.,data = train)
  svm_predict<-predict(svm_model,test[,-1])
  svm_error[i]<-1-(sum(svm_predict == test[,1] ) / length(svm_predict))
  
  
  bayes_model<-naiveBayes(M~.,data = train)
  bayes_predict<-predict(bayes_model,test[,-1])
  bayes_error[i]<-1-(sum(bayes_predict == test[,1] ) / length(bayes_predict))
  
  logistic_model<-glm(M~.,family=binomial(link='logit'),data=train)
  logistic_predict<-predict(logistic_model,test[,-1])
  logistic_error[i]<-1-(sum(logistic_predict == test[,1] ) / length(logistic_predict))
  
}
# Compare the performance of the different classifiers using ANOVA test (see ?aov)
errors <- c(c50_error,svm_error,bayes_error,logistic_error)
errors
classifiers <- c(rep("decision_tree",10),rep("SVM",10),rep("Naive_bayes",10),rep("Logistic_regression",10))
ec <- data.frame(errors,classifiers)

anova_result <- aov(errors~classifiers,data=ec)
print(summary(anova_result))
anova_result

#p-value <2e-16 which is less than 0.05 thus we reject null hypothesis

# *****************************************
# Part 3: Wilcoxon Signed Rank test
# *****************************************

# Load the following data sets,
# 1. Iris 
i_data<-read.csv("./datasets/Iris_data.txt")


# 2. Ecoli 
e_data<-read.csv("./datasets/Ecoli_data.csv")


# 3. Wisconsin Breast Cancer
b_data<-read.csv("./datasets/Wisconsin_Breast_Cancer_data.txt")
b_data<-b_data[,-1]

# 4. Glass
g_data<-read.csv("./datasets/Glass_data.txt")


# 5. Yeast
y_data<-read.csv("./datasets/Yeast_data.csv")

# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
set.seed(1430)
i_data <- i_data[sample(nrow(i_data)),]
i_data
k <- 10 

i_folds <- cvFolds(NROW(i_data), K=k)

set.seed(1430)
e_data <- e_data[sample(nrow(e_data)),]
head(e_data)
k <- 10 

e_folds <- cvFolds(NROW(e_data), K=k)

set.seed(1430)
b_data <- b_data[sample(nrow(b_data)),]
head(b_data)
k <- 10 

b_folds <- cvFolds(NROW(b_data), K=k)

set.seed(1430)
g_data <- g_data[sample(nrow(g_data)),]
head(g_data)
k <- 10 

g_folds <- cvFolds(NROW(g_data), K=k)

set.seed(1430)
y_data <- y_data[sample(nrow(y_data)),]
y_data
k <- 10 

y_folds <- cvFolds(NROW(y_data), K=k)


# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)

i_c50_error<-0
i_svm_error<-0

b_c50_error<-0
b_svm_error<-0

e_c50_error<-0
e_svm_error<-0

g_c50_error<-0
g_svm_error<-0

y_c50_error<-0
y_svm_error<-0

for(i in 1:k){
  train <- i_data[i_folds$subsets[i_folds$which != i], ] #Set the training set
  test <- i_data[i_folds$subsets[i_folds$which == i], ] #Set the validation set
  
  i_c50_decision_tree<-C5.0(x=train[,-5],y=train$Iris.setosa)
  i_c50_predict<-predict(i_c50_decision_tree,test[,-5])
  i_c50_error[i]<-1-(sum(i_c50_predict == test[,5] ) / length(i_c50_predict))
  
  i_svm_model<-svm(Iris.setosa~.,data = train)
  i_svm_predict<-predict(i_svm_model,test[,-5])
  i_svm_error[i]<-1-(sum(i_svm_predict == test[,5] ) / length(i_svm_predict))

  train <- b_data[b_folds$subsets[b_folds$which != i], ] #Set the training set
  test <- b_data[b_folds$subsets[b_folds$which == i], ] #Set the validation set
  
  b_c50_decision_tree<-C5.0(x=train[,-1],y=train$M)
  b_c50_predict<-predict(b_c50_decision_tree,test[,-1])
  b_c50_error[i]<-1-(sum(b_c50_predict == test[,1] ) / length(b_c50_predict))
  
  b_svm_model<-ksvm(M~.,data = train)
  b_svm_predict<-predict(b_svm_model,test[,-1])
  b_svm_error[i]<-1-(sum(b_svm_predict == test[,1] ) / length(b_svm_predict))
  
  train <- e_data[e_folds$subsets[e_folds$which != i], ] #Set the training set
  test <- e_data[e_folds$subsets[e_folds$which == i], ] #Set the validation set
  
  e_c50_decision_tree<-C5.0(x=train[,-9],y=train$cp)
  e_c50_predict<-predict(e_c50_decision_tree,test[,-9])
  e_c50_error[i]<-1-(sum(e_c50_predict == test[,9] ) / length(e_c50_predict))
  
  e_svm_model<-ksvm(cp~.,data = train)
  e_svm_predict<-predict(e_svm_model,test[,-9])
  e_svm_error[i]<-1-(sum(e_svm_predict == test[,9] ) / length(e_svm_predict))

  train <- g_data[g_folds$subsets[g_folds$which != i], ] #Set the training set
  test <- g_data[g_folds$subsets[g_folds$which == i], ] #Set the validation set
  
  g_c50_decision_tree<-C5.0(x=train[,-11],y=as.factor(train$X1.1))
  g_c50_predict<-predict(g_c50_decision_tree,test[,-11])
  g_c50_error[i]<-1-(sum(g_c50_predict == test[,11] ) / length(g_c50_predict))
  
  g_svm_model<-ksvm(X1.1~.,data = train)
  g_svm_predict<-predict(g_svm_model,test[,-11])
  g_svm_error[i]<-1-(sum(g_svm_predict == test[,11] ) / length(g_svm_predict))
  
  train <- y_data[y_folds$subsets[y_folds$which != i], ] #Set the training set
  test <- y_data[y_folds$subsets[y_folds$which == i], ] #Set the validation set
  
  y_c50_decision_tree<-C5.0(x=train[,-10],y=train$MIT)
  y_c50_predict<-predict(y_c50_decision_tree,test[,-10])
  y_c50_error[i]<-1-(sum(y_c50_predict == test[,10] ) / length(y_c50_predict))
  
  y_svm_model<-ksvm(MIT~.,data = train)
  y_svm_predict<-predict(y_svm_model,test[,-10])
  y_svm_error[i]<-1-(sum(y_svm_predict == test[,10] ) / length(y_svm_predict))
  
}

# Make predictions on the test set and calculate the error percentages made by the trained models

# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)

mean(i_c50_error)
mean(i_svm_error)

mean(b_c50_error)
mean(b_svm_error)

mean(e_c50_error)
mean(e_svm_error)

mean(g_c50_error)
mean(g_svm_error)

mean(y_c50_error)
mean(y_svm_error)

c50_error=c(mean(i_c50_error),mean(b_c50_error),mean(e_c50_error),mean(g_c50_error),mean(y_c50_error))
svm_error = c(mean(i_svm_error),mean(b_svm_error),mean(e_svm_error),mean(g_svm_error),mean(y_svm_error))

wilcox_result=wilcox.test(c50_error,svm_error,paired=TRUE)
print(wilcox_result)
#p-value is 0.8551 > 0.05 thus we do not reject the null hyputhesis
