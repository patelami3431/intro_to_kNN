# author: "Ami Patel"
# title: "kNN Classification"
# date: "03/03/2019"

###############################################################################################################

# For data exploration,
data("iris")
str(iris)               # provides the structure of iris data
head(iris)              # gives the first 6 observations
names(iris)             # shows the column names of the iris data
summary(iris$Species)   # Displays the number of setosa, versicolor and virginica

###### PLOT 1 #############################################################

# To see how well can Sepal.Length, Sepal.Width, Petal.Length and Petal.Width help us classify the Species
par(mfrow = c(1,2))
plot(iris$Sepal.Length, iris$Sepal.Width, pch=23, bg=c("red","green3","blue")[unclass(iris$Species)], main="PLOT 1")
plot(iris$Petal.Length, iris$Petal.Width, pch=23, bg=c("red","green3","blue")[unclass(iris$Species)], main="PLOT 2")


###### Perform the kNN classification on the data #############################################################
startTime <- proc.time()                                       # stores the starts time

set.seed(1234)                                                 # sets the seed for reproducibility of results
i <- sample(1:nrow(iris), 0.80 * nrow(iris), replace = FALSE)  # randomly chooses 80% of the rows to store in i
train <- iris[i,]                                              # assigns 80% rows to train dataset
test <- iris[-i,]                                              # assigns the remaining 20% of the rows to test dataset

library(class)
trainLabels = train$Species                                    # moved Species label to the train labels
testLabels = test$Species

########## train the algorithm and classify the test points

pred_kNN = knn(train = train[,1:4],test = test[,1:4],cl = trainLabels, k = 3)

acc_kNN <- mean(pred_kNN == testLabels) * 100                  # computes the accuracy of kNN
print(paste("kNN accuracy is ", acc_kNN, "%"))
table(testLabels, pred_kNN)                                    # prints the confusion matrix for the kNN

endTime <- proc.time()                                         # stores the end time

runTime <- endTime - startTime                                 # calculate the difference to get the elapsed time
print(paste("The run time for R script is ", runTime[[3]] * 1000, "ms"))

###### PLOT 2 #############################################################################

# plot the test points to see how well the Species were classified
library(plyr)
library(ggplot2)
plot.df = data.frame(x = test$Petal.Length, y = test$Petal.Width, predicted = pred_kNN)

find_hull = function(d) d[chull(d$x, d$y), ]      # convex function to see the nearby close points
boundary = ddply(plot.df, .variables = "predicted", .fun = find_hull)

ggplot(plot.df, aes(test$Petal.Length, test$Petal.Width, color = predicted, fill = predicted)) + geom_polygon(data = boundary, aes(x,y), alpha = 0.2) + geom_point(size = 2)

###### CREATING THE .CSV FILES  #############################################################
write.csv(train, file = "train.csv", row.names = FALSE)       # writes the train data to train.csv, while ignoring the row names
write.csv(test, file = "test.csv", row.names = FALSE)         # writes the test data to test.csv, while ignoring the row names 



