---
  title: "Leaf Fresh Weight Prediction using Random Forest"
author: "Jonathan Cárdenas"
date: "2023-04-04"
output: html_document
---
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(BBmisc)
library(tidyverse)
library(dplyr)
library(readr)
library(caret)
library(ggridges)
library(reshape2)
library(lubridate)
library(RColorBrewer)
library(corrplot)
library(GGally)
library(party)
library(MASS)
library(randomForest)
```

```{r}
raw_data<- read.csv("harvest_complete.csv")
raw_data$treatment <-as.factor(with(raw_data, ifelse(tray_id == '1', '280 ppm',
                                                     ifelse(tray_id == 2, '160 ppm', '80 ppm'))))
raw_data <- raw_data %>% relocate(treatment)
raw_data$new_feature <- raw_data$plant_area * raw_data$height_mm
```


Create 3 dataframes for predictors, response variable and combination of both
```{r}
# normalize img predictors
img_predictors <- raw_data %>% dplyr::select(13:26) #1
predictors_norm <- normalize(img_predictors, method = 'standardize')

# extract LFW from raw data
LFW_date<- raw_data %>% dplyr::select(c('LFW_g','date'))

# combine normalized predictors and raw data. 
img_LFW <- bind_cols(predictors_norm,LFW_date)
```

Select a random sample of 33 entries for the last two harvest:
  ```{r}
set.seed(12)
harvest_1 <-img_LFW %>% filter(date == '02/02/2023') 

harvest_2 <- img_LFW %>% filter(date == '09/02/2023') %>% sample_n(33)

harvest_3 <-  img_LFW %>% filter(date == '23/02/2023') %>% sample_n(33)

harvest_sample <- bind_rows(harvest_1,harvest_2,harvest_3) %>% dplyr::select(-date)
```

Select a representative number of features according to BIC analysis performed in the markdown for feature selection. Select the features using dplyr select. 
```{r}
# BIC  features
new_dataset <- harvest_sample %>% dplyr::select(height_mm,new_feature,plant_area,plant_ellipse_major_axis,plant_ellipse_minor_axis, plant_convex_hull_area, LFW_g)


```

Separate data in train and test data: 
  ```{r}
# Use a random forest model to predict LFW, this time splitting the dataset. 
set.seed(79)
train_id<- sample(1:nrow(new_dataset),0.7*nrow(new_dataset))
train_set<- new_dataset[train_id,]
test_set<- new_dataset[-train_id,]
```
Fit random forest model:
  The following chart shows the models summary. Is important to mention here that the models summary give us an out-of-bag MSE (for values used to train the model).

```{r}
RF_model <- randomForest(LFW_g~., data = train_set, ntree = 500, importance= TRUE)
RF_model
```
A variable importance plot is showing us which of the predictors used (variables) were most important in predicting the response variable. With this we can identify the key features that are driving the model's predictions. In this plot, we are using the increase in mean squared error (the greater the increase the more important the variable is for the models predictive performance). Notice the longest bars and the highest percentages. 

```{r}
var_imp <- importance(RF_model, type = 1)
var_imp_df <- data.frame(var_imp)
var_imp_df$variable <- rownames(var_imp_df)

ggplot(var_imp_df, aes(x = reorder(variable,X.IncMSE), y = X.IncMSE)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  xlab("Variable") +
  ylab("Mean Decrease Gini") +
  ggtitle("Variable Importance Plot")
```


```{r}
varImpPlot(RF_model, sort = TRUE, n.var = 5, main = "Variable Importance Plot")
```


From now on, we will use the test set to perform predictions and to calculate all the performance statistics. 
```{r}
RF_prediction <- predict(RF_model, newdata = test_set)
```
Notice that the R-squared value obtained below is not the same as the R-quared given by the model's summary. For this R-squared we are correlating the predicted values and the labels from the test set.

MSE displayed is not the same as the summary MSE neither. The MSE we are calculating here is mean squared error of the model's prediction on the testing data. This metric measures how well the model generalizes to new, unseen data. 

While the two MSE values are related, they are not the same. The out-of-bag MSE gives an estimate of the model's generalization performance based on the training data, while the MSE on the test set provides an actual measure of how well the model performs on new, unseen data.
```{r}
# Evaluate the model performance: On new, unseen data. 
MAE <- mean(abs(RF_prediction - test_set$LFW_g))
MSE <- mean((RF_prediction - test_set$LFW_g)^2)
R_squared <- cor(RF_prediction,test_set$LFW_g)^2
RMSE<- RMSE(RF_prediction,test_set$LFW_g)

# Print the performance metrics
cat("Mean Absolute Error (MAE): ", MAE, "\n")
cat("Mean Squared Error (MSE): ", MSE, "\n")
cat("R-squared: ", R_squared, "\n")
cat("RMSE (Prediction): ", RMSE, "\n")


ggplot(data.frame(actual = test_set$LFW_g, predicted = RF_prediction), aes(x = actual, y = predicted)) + 
  theme_bw()+
  geom_point(color = "black", fill = "red",shape = 21, size = 2, stroke = 1) +
  geom_abline(slope = 1, intercept = 0, color = "black", size =0.8) +
  theme_light() +
  theme(plot.title = element_text(size=16,hjust = 0.5, face="bold"),
        axis.text = element_text(size = 7,color="black"),
        axis.title.x = element_text(size=12,face="bold"),
        axis.title = element_text(size = 12, face="bold"),
        legend.position = "top",
        legend.background = element_rect(fill="white",
                                         size=1, linetype="solid", 
                                         colour ="black"),
        legend.text = element_text(size = 10),
        legend.title = element_text(size = 10,
                                    face = "bold"),
        strip.background = element_rect(color="black", fill="green4", 
                                        size=1.5, linetype="solid"),
        strip.text.x = element_text(
          size = 12, color = "white", face = "bold.italic"))+
  labs(x = "Actual LFW (g)", 
       y = "Predicted LFW (g)", 
       title = "Accuracy: Random Forest Regression Model") +
  annotate("text", x = 20, y = 130, label = paste("R-squared: ", round(R_squared, 3), "\n", "RMSE: ", round(RMSE, 3)), size = 4, color = "black")

# Using the model with the best 6 predictors to estimate LFW for each harvest day.


H3_best <- harvest_3 %>% dplyr::select(height_mm,new_feature,plant_area,plant_ellipse_major_axis,plant_ellipse_minor_axis, plant_convex_hull_area, LFW_g)


RF_prediction_H3 <- predict(RF_model, newdata = H3_best)
RF_prediction_H3

# Evaluate the model performance: On new, unseen data. 
MAE <- mean(abs(RF_prediction_H3 - H3_best$LFW_g))
MSE <- mean((RF_prediction_H3 - H3_best$LFW_g)^2)
R_squared <- cor(RF_prediction_H3,H3_best$LFW_g)^2
RMSE<- RMSE(RF_prediction_H3,H3_best$LFW_g)

# Print the performance metrics
cat("Mean Absolute Error (MAE): ", MAE, "\n")
cat("Mean Squared Error (MSE): ", MSE, "\n")
cat("R-squared: ", R_squared, "\n")
cat("RMSE (Prediction): ", RMSE, "\n")

ggplot(data.frame(actual = H3_best$LFW_g, predicted = RF_prediction_H3), aes(x = actual, y = predicted)) + 
  geom_point(color = "black", fill = "red",shape = 21, size = 2, stroke = 1) +
  geom_abline(slope = 1, intercept = 0, color = "black", size =0.8) +
  theme_light() +
  theme(plot.title = element_text(size=20,hjust = 0.5, face="bold"),
        axis.text = element_text(size = 10,color="black"),
        axis.title.x = element_text(size=16,face="bold"),
        axis.title = element_text(size = 16, face="bold"),
        legend.position = "top",
        legend.background = element_rect(fill="white",
                                         size=1, linetype="solid", 
                                         colour ="black"),
        legend.text = element_text(size = 10),
        legend.title = element_text(size = 10,
                                    face = "bold"),
        strip.background = element_rect(color="black", fill="green4", 
                                        size=1.5, linetype="solid"),
        strip.text.x = element_text(
          size = 12, color = "white", face = "bold.italic"))+
  labs(x = "Actual LFW (g)", 
       y = "Predicted LFW (g)", 
       title = "Accuracy: Random Forest Regression Model") +
  annotate("text", x = 80, y = 130, label = paste("R-squared: ", round(R_squared, 3), "\n", "RMSE: ", round(RMSE, 3)), size = 5, color = "black")

H2_best <- harvest_2 %>% dplyr::select(height_mm,new_feature,plant_area,plant_ellipse_major_axis,plant_ellipse_minor_axis, plant_convex_hull_area, LFW_g)


RF_prediction_H2 <- predict(RF_model, newdata = H2_best)
RF_prediction_H2



# Evaluate the model performance: On new, unseen data. 
MAE <- mean(abs(RF_prediction_H2 - H2_best$LFW_g))
MSE <- mean((RF_prediction_H2 - H2_best$LFW_g)^2)
R_squared <- cor(RF_prediction_H2,H2_best$LFW_g)^2
RMSE<- RMSE(RF_prediction_H2,H2_best$LFW_g)

# Print the performance metrics
cat("Mean Absolute Error (MAE): ", MAE, "\n")
cat("Mean Squared Error (MSE): ", MSE, "\n")
cat("R-squared: ", R_squared, "\n")
cat("RMSE (Prediction): ", RMSE, "\n")

ggplot(data.frame(actual = H2_best$LFW_g, predicted = RF_prediction_H2), aes(x = actual, y = predicted)) + 
  geom_point(color = "black", fill = "red",shape = 21, size = 2, stroke = 1) +
  geom_abline(slope = 1, intercept = 0, color = "black", size =0.8) +
  theme_light() +
  theme(plot.title = element_text(size=18,hjust = 0.5, face="bold"),
        axis.text = element_text(size = 10,color="black"),
        axis.title.x = element_text(size=16,face="bold"),
        axis.title = element_text(size = 16, face="bold"),
        legend.position = "top",
        legend.background = element_rect(fill="white",
                                         size=1, linetype="solid", 
                                         colour ="black"),
        legend.text = element_text(size = 10),
        legend.title = element_text(size = 10,
                                    face = "bold"),
        strip.background = element_rect(color="black", fill="green4", 
                                        size=1.5, linetype="solid"),
        strip.text.x = element_text(
          size = 12, color = "white", face = "bold.italic"))+
  labs(x = "Actual LFW (g)", 
       y = "Predicted LFW (g)", 
       title = "Accuracy: Random Forest Regression Model",
       subtitle = "Model performance on unseen data (test  set) for second harvest day ") +
  annotate("text", x = 5, y = 15, label = paste("R-squared: ", round(R_squared, 3), "\n", "RMSE: ", round(RMSE, 3)), size = 5, color = "black")



ggsave("ggplotsave.jpg",width = 6,height = 5,units = c("in"))
