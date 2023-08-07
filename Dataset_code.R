setwd("/Users/hanis/Downloads/code_artefact_x20246862/code_artefact_x20246862")

# Import libraries
library(dplyr)
library(caTools)
library(caret)
library(e1071)
library(gmodels)
library(ROCR)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(ROSE)
library(tidyr)
library(multiROC)
library(class)
library(OneR)
library(tidyverse)
library(ggstatsplot)
library(corrplot)
library(modelr)
library(plotly)
library(cluster)
library(ggfortify)

# Dataset 1

#Load data
hotelData <- read.csv('hotel_bookings.csv')

# Convert data into data frame
df <- as.data.frame(hotelData)

# Inspect the variables
print(df)

# Data Cleaning

# Drop company column as it has NULL values
df1 <- subset(df, select= -c(company))
print(df1)

# Omit NULL values in the agent column
df2 <- subset(df1, agent != "NULL")
print(df2)

# Remove negative values in adr column
df3 <- subset(df2, adr != "-6.38")
  
# Check for missing values
sapply(df3, function(x) sum(is.na(x))) 

# Remove missing values
df4 <- na.omit(df3)
print(df4)

# Check any duplicated values
duplicated(df4)

# extract duplicated elements
df4[duplicated(df4)]

# Remove duplicated elements
df5 <- df4[!duplicated(df4), ]

# inspect variables
print(df5)
str(df5)

# Data Transformation

# convert target variable into factor
df5$is_canceled <- as.factor(df5$is_canceled)

# Check any class imbalance pertaining to is_cancelled variable
summary(df5$is_canceled) 

# Visualize class imbalance
barplot(prop.table(table(df5$is_canceled)),
        col = rainbow(2),
        ylim = c(0, 0.7),
        main = "Class Distribution")

# split into train and test set
set.seed(100)
split <- sample.split(df5$is_canceled, SplitRatio = 0.7)
trainData_1 <- subset(df5, split == TRUE)
testData_1 <- subset(df5, split == FALSE)

# Predictive model with class imbalance

# Naïve Bayes Classifier

#isolate the target variable
target_variable <- trainData_1$is_canceled

# remove target variable from the training set
train_rm <- subset(trainData_1, select = -c(is_canceled))

classifier_1 <- naiveBayes(train_rm, target_variable)
predict_1 <- predict(classifier_1, testData_1)

# confusion matrix
confusionMatrix(predict_1, testData_1$is_canceled) # Accuracy 69.44%

# Predictive model without class imbalance (resampling)

# under-sampling
under <- ovun.sample(is_canceled~., 
                     data=trainData_1,
                     seed = 123,
                     method = 'under')$data
# over-sampling
over <- ovun.sample(is_canceled~., 
                     data=trainData_1,
                     seed = 123,
                     method = 'over')$data

table(under$is_canceled)
table(over$is_canceled)

# Naïve Bayes Classifier for under sampling

#isolate the target variable
target_variable_un <- under$is_canceled

# remove target variable from the training set
train_rm_un <- subset(under, select = -c(is_canceled))

classifier_un <- naiveBayes(train_rm_un, target_variable_un)
predict_un <- predict(classifier_un, testData_1)

# confusion matrix
confusionMatrix(predict_un, testData_1$is_canceled) # Accuracy 63.38%


# Naïve Bayes Classifier for over sampling

#isolate the target variable
target_variable_ov <- over$is_canceled

# remove target variable from the training set
train_rm_ov <- subset(over, select = -c(is_canceled))

classifier_ov <- naiveBayes(train_rm_ov, target_variable_ov)
predict_ov <- predict(classifier_ov, testData_1)

# confusion matrix
confusionMatrix(predict_ov, testData_1$is_canceled) # Accuracy 61.53%

# Plot multiple ROC curves
roc1 <- roc(predict_1, as.numeric(testData_1$is_canceled))
roc2 <- roc(predict_un, as.numeric(testData_1$is_canceled))
roc3 <- roc(predict_ov, as.numeric(testData_1$is_canceled))

plot(roc1, col = 1, lty = 2, main = "ROC", print.auc = TRUE, print.auc.y = .4)
plot(roc2, col = 2, lty = 2, add = TRUE, print.auc = TRUE, print.auc.y = .3)
plot(roc3, col = 3, lty = 2, add = TRUE, print.auc = TRUE, print.auc.y = .2)

legend("bottomright", c("normal", "under", "over"), lty=1, 
       col = c("black", "pink", "green"))

# Random Forest
rf_1 <- randomForest(is_canceled~., data = under) #using under sampled data
print(rf_1)
plot(rf_1)
# prediction
predict_rf <- predict(rf_1, testData_1)
confusionMatrix(predict_rf, testData_1$is_canceled) # Accuracy 100%

# ROC to compare NVB with Random Forest
roc1 <- roc(predict_rf, as.numeric(testData_1$is_canceled))
roc2 <- roc(predict_un, as.numeric(testData_1$is_canceled))

plot(roc1, col = 1, lty = 2, main = "ROC", print.auc = TRUE, print.auc.y = .4)
plot(roc2, col = 2, lty = 2, add = TRUE, print.auc = TRUE, print.auc.y = .3)

legend("top", c("RF", "NBV"), lty=1, 
       col = c("black", "pink"))

# Dataset 2

musicData <- read.csv('music_genre.csv')

# Convert data into data frame
df <- as.data.frame(musicData)

# Inspect the variables
print(df)

# Data Cleaning

# Check for missing values
sapply(df, function(x) sum(is.na(x))) 

# Remove missing values
df1 <- na.omit(df)

# Drop irrelevant features
df2 <- subset(df1, select= -c(instance_id, obtained_date, artist_name, track_name, key, mode))

# remove negative value in duration and '?' value in tempo
df3 <- subset(df2, duration_ms != -1)
df4 <- subset(df3, tempo != '?')

# Check any duplicated values
duplicated(df4)

# Remove duplicated elements
df5 <- df4[!duplicated(df4), ]

str(df5)

# Data Transformation

# isolate target variable
no_cat <- subset(df5, select = -c(music_genre))

#convert variables as numeric for correlation matrix
no_cat$popularity <- as.numeric(no_cat$popularity)
no_cat$duration_ms <- as.numeric(no_cat$duration_ms)
no_cat$tempo <- as.numeric(no_cat$tempo)

str(no_cat)

cor1 <- cor(no_cat)
corrplot(cor1, method = 'number')

# create normalize function
normalize <- function(x) {(x -min(x))/(max(x)-min(x))}

# convert variable as numeric for modelling
df5$popularity <- as.numeric(df5$popularity)
df5$duration_ms <- as.numeric(df5$duration_ms)
df5$tempo <- as.numeric(df5$tempo)

# run normalization on predictors
genre_norm <- as.data.frame(lapply(df5[,c(1:11)], normalize))

# Split into test and train set 
set.seed(100)
split <- sample.split(genre_norm, SplitRatio = 0.7)
trainData_2 <- subset(genre_norm, split == TRUE)
testData_2 <- subset(genre_norm, split == FALSE)

#extract target variable of train set to be used as 'cl' argument in knn function.
train_target <- df5[split,12]

# same in test set to measure the accuracy
test_target <- df5[!split, 12]


# Predictive Modelling

# KNN
pred_knn <- knn(trainData_2, testData_2, cl=train_target, k=20) # Initially k=10 (Failed method)
pred_knn

df_pred <- data.frame(test_target, pred_knn) # comparison
View(df_pred)

#Evaluate the model performance
table <- table(test_target,pred_knn)
View(table)

confusionMatrix(pred_knn, test_target) 

# Predictive modelling

# Support Vector Machine (SVM)

train_target <- as.factor(train_target)

svm_1 <- svm(train_target~., data = trainData_2, type='C-classification', kernel='radial')
summary(svm_1)

#Prediction
svm_pred <- predict(svm_1, newdata = testData_2, type="class")

test_target <- as.factor(test_target)

confusionMatrix(svm_pred, test_target) 

# conversion for auc curve svm
auc_test <- as.numeric(test_target)
auc_svm_pred <- as.numeric(svm_pred)

# conversion for auc curve knn
auc_knn_pred <- as.numeric(pred_knn)

# ROC AUC 

# SVM
roc1 <- multiclass.roc(auc_svm_pred, auc_test)

# KNN
roc2 <- multiclass.roc(auc_knn_pred, auc_test)

roc1$auc # 0.7406
roc2$auc # 0.7047

# SVM performed better than KNN

# Dataset 3

bbData <- read.csv('airbnb.csv')

# Convert data into data frame
df <- as.data.frame(bbData)

# Inspect the variables
print(df)
str(df)

# Data Cleaning

# Drop unhelpful columns
df1 <- subset(df, select= -c(host_name, name, host_id, id, license, number_of_reviews_ltm, last_review, latitude, longitude))
print(df1)

# Check for missing values
sapply(df1, function(x) sum(is.na(x)))

# Fill NA values in reviews per month with 0's as the features important for analysis
df1$reviews_per_month[is.na(df1$reviews_per_month)] <- 0

# check outliers
df1$price <- as.numeric(df1$price)

# create boxplot for certain variables
boxplot(price~neighbourhood_group,
        data=df1,
        col="steelblue",
        border="black")

# Remove extreme outliers using interquartile range
Q1 <- quantile(df1$price, .25)
Q3 <- quantile(df1$price, .75)
IQR <- IQR(df1$price)

no_outliers <- subset(df1, df1$price > (Q1 - 1.5*IQR) & df1$price < (Q3 + 1.5*IQR))
dim(no_outliers)

class(no_outliers)

#check again using boxplot
boxplot(price~neighbourhood_group,
        data=no_outliers,
        col="steelblue",
        border="black"
)

# inspect variables
print(no_outliers)
str(no_outliers)

# Check any duplicated values
duplicated(no_outliers)

# Remove duplicated elements
no_outliers <- no_outliers[!duplicated(no_outliers), ]

# Data Transformation

# one-hot encode categorical variables
result1 <- no_outliers %>% mutate(value = 1)  %>% spread(neighbourhood_group, value, fill = 0 )
result2 <- result1 %>% mutate(value = 1)  %>% spread(room_type, value,  fill = 0 )

# rename c=spaces with '_'
colnames(result2)[11] <- "Staten_Island"
colnames(result2)[12] <- "Entire_home_apt"
colnames(result2)[13] <- "Hotel_room"
colnames(result2)[14] <- "Private_room"
colnames(result2)[15] <- "Shared_room"

# one-hot encode categorical variable
#result3 <- result2  %>% mutate(value = 1)  %>% spread(neighbourhood, value,  fill = 0 )

result4 <- subset(result2, select= -c(neighbourhood))

result4 <- subset(result4, select= -c(minimum_nights, number_of_reviews, calculated_host_listings_count,availability_365))

result4 <- subset(result4, select= -c(reviews_per_month))

corrplot(cor(df2),
         method = "number",
         type = "upper", # show only upper side
         addCoef.col = 1,    # Change font size of correlation coefficients
         number.cex = 0.75
)


corrplot(cor(result4),
         method = "number",
         type = "upper", # show only upper side
         addCoef.col = 1,    # Change font size of correlation coefficients
         number.cex = 0.75,
         cl.cex = 0.5
)


# split to train and test
#set.seed(100)
#split_3 <- sample.split(result$price, SplitRatio = 0.7)
#trainData_3 <- subset(result3, split_3 == TRUE)
#testData_3 <- subset(result3, split_3 == FALSE)

# split to train and test
set.seed(42)
split_3 <- sample.split(result4$price, SplitRatio = 0.8)
trainData_3 <- subset(result4, split_3 == TRUE)
testData_3 <- subset(result4, split_3 == FALSE)

# Linear Regression

model_lm <- lm(price ~ Manhattan + Hotel_room + Bronx + Shared_room, data=trainData_3)
summary(model_lm)

# Get residuals
lm_residuals <- as.data.frame(residuals(model_lm))

# Visualize residuals
ggplot(lm_residuals, aes(residuals(model_lm))) +
  geom_histogram(fill = "#0099f9", color = "black") +
  theme_classic() +
  labs(title = "Residuals plot")

# prediction
pred_lm <- predict(model_lm, testData_3)
pred_lm
#summary
summary(pred_lm)

# actual vs predicted

modelEval <- cbind(testData_3, pred_lm)

# Evaluate model

mse <- mean((modelEval$Actual - modelEval$Predicted)**2)
mse
rmse <- sqrt(mse) # 57.87%
rmse

# Decision Tree

tree <- rpart(price ~ Manhattan + Hotel_room + Bronx + Shared_room, data= trainData_3)
summary(tree)
rpart.plot(tree)

printcp(tree)

# Predict

pred_tree <- predict(tree, testData_3)
print(pred_tree)


treeEval <- cbind(testData_3,pred_tree)

# evaluate

# MSE
mse_2 <- mean((trainData_3$price - pred_tree)^2)
mse_2
# RMSE
rmse_2 <- sqrt(mean((trainData_3$price - pred_tree)^2)) # 59.81%
rmse_2

#############################

# Predicting new AirBnB rental prices in NYC using Multilayer Perceptron & Feature selection using Recursive Feature Elimination (RFE)

# Load necessary libraries
library(caret)
library(keras)
library(dplyr)

# Load the dataset
bbData <- read.csv('airbnb.csv')

# Extract features and target (price)
numeric_features <- as.matrix(bbData[, c('minimum_nights', 'availability_365')])
categorical_feature <- bbData[, c('neighbourhood_group', 'room_type')]
prices <- bbData$price

# Preprocess the data (feature scaling for numeric features)
#feature_means <- colMeans(numeric_features)
#feature_stds <- apply(numeric_features, 2, sd)
#numeric_features <- scale(numeric_features, center = feature_means, scale = feature_stds)

#categorical_features <- do.call(cbind, new_factors)
#rownames(categorical_features) <- NULL  # Reset row names

# One-hot encode the categorical feature
dummy_data <- dummyVars("~.", data.frame(categorical_feature))
categorical_features <- predict(dummy_data, newdata = data.frame(categorical_feature))

# Combine numeric and categorical features
features <- cbind(numeric_features, categorical_features)

# Define the feature selection control parameters
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "cv", number = 5)

# Perform Recursive Feature Elimination (RFE)
selected_features <- rfe(features, prices, 
                         sizes = c(1:ncol(features)), 
                         rfeControl = ctrl)

# Get the selected feature indices
selected_indices <- as.factor(selected_features$optVariables)
selected_features_names <- colnames(features)[selected_indices]

print(selected_features_names) # to view the selected features


# Filter the data based on selected features
selected_features <- features[, selected_indices]

# Split data into training and testing sets
train_ratio <- 0.8
num_train_samples <- round(nrow(selected_features) * train_ratio)

x_train <- selected_features[1:num_train_samples, ]
y_train <- prices[1:num_train_samples]

x_test <- selected_features[(num_train_samples + 1):nrow(selected_features), ]
y_test <- prices[(num_train_samples + 1):nrow(selected_features)]

# Define the MLP model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(selected_features)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)  # Output layer with a single neuron for price prediction

# Compile the model
model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(),
  metrics = c("mean_absolute_error")
)

# Train the model with selected features
history <- model %>% fit(
  x = x_train,
  y = y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate the model with selected features
evaluation <- model %>% evaluate(x_test, y_test)
print(evaluation)

# Make predictions for the selected features using the trained model
predicted_price <- model %>% predict(list(dense_2_input=
                                            as.matrix(selected_features)))

# Convert predicted_price to a dataframe with selected_features as columns
predicted_df <- data.frame(predicted_price)

# Convert 'selected_features' to a dataframe with appropriate column names
selected_features_df <- data.frame(selected_features)
colnames(selected_features_df) <- colnames(selected_features)

# Combine 'selected_features_df' with 'predicted_df'
result_df <- cbind(selected_features_df, Predicted_Price = predicted_df)

# Print the final dataframe with selected feature names and predicted prices
print("Final DataFrame:")
print(result_df)
