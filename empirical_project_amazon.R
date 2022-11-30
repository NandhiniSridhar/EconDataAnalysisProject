install.packages("caret")
install.packages("fastDummies")
#https://www.kaggle.com/datasets/surajjha101/myntra-reviews-on-women-dresses-comprehensive

# setwd("/Users/NandhiniSridhar/Desktop/142_empirical_project")
setwd('.')
getwd()

reviews <- read.csv("WomenDressesReviewsDataset.csv")
head(reviews)
colnames(reviews)
attach(reviews)

#data cleanup
#creating dummy variables
library(fastDummies)
library(stringr)
unique(reviews$division_name)
unique(reviews$class_name)
unique(reviews$rating)

#drop any rows with null values
reviews <- na.omit(reviews)

#create dummies for categorical
reviews <- dummy_cols(reviews, select_columns = "division_name")
reviews <- dummy_cols(reviews, select_columns = "department_name")
reviews <- dummy_cols(reviews, select_columns = "class_name")
reviews <- dummy_cols(reviews, select_columns = "rating")

#drop one of the dummy columns to avoid perfect correlation
reviews <- subset(reviews, select = -c(class_name_, class_name_Chemises, division_name_, division_name_General, rating_2, department_name_Bottoms))

#create num_words column
reviews$num_words_review <- str_count(reviews$review_text)

summary(reviews)


#initial plots
plot(rating ~ alike_feedback_count, data=reviews)
boxplot(rating, main = "Ratings")
boxplot(alike_feedback_count, main = "Alike Feedback Count")


#split into training and testing sets
train_num <- trunc(0.8 * nrow(reviews))

train = reviews$s.no < train_num
reviews_train = reviews[train,]
reviews_test = reviews[!train, ]
rec_test = reviews$recommend_index[!train]


# Logistic regressions with chosen variables, polynomial transformations, and step functions
set.seed(0)

log_reg <- glm(recommend_index ~ rating_1 + rating_3 + rating_4 + rating_5 + num_words_review + alike_feedback_count, data = reviews, family = "binomial", subset = train)
log_reg 
summary(log_reg)
pred1 <- predict(log_reg, newdata = reviews_test)
mse1 <- mean((rec_test - pred1)^2)
mse1

set.seed(0)

log_reg2 <- glm(recommend_index ~ alike_feedback_count + num_words_review + poly(rating, 2), data = reviews, family = "binomial")
log_reg2 
summary(log_reg2)
pred2 <- predict(log_reg2, newdata = reviews_test)
mse2 <- mean((rec_test - pred2)^2)
mse2

set.seed(0)

log_reg_step <- glm(recommend_index ~ cut(rating, 4), family='binomial', data=reviews)
pred3 <- predict(log_reg_step, newdata = reviews_test)
mse3 <- mean((rec_test - pred3)^2)
mse3

set.seed(0)

#ridge and lasso regression
install.packages("glmnet")
library(glmnet)
set.seed(0)

#create a model matrix that only includes columns with numeric data or encoded categorical data
#remove all text data and non-encoded categorical data
reviews_numerical <- subset(reviews, select = -c(division_name, department_name, class_name, title, review_text, s.no, clothing_id, rating))
names(reviews_numerical) <- sub(" ", "", names(reviews_numerical))
train = reviews$s.no < train_num
reviews_numerical_train = reviews_numerical[train,]
reviews_numerical_test = reviews_numerical[!train, ]
reviews_numerical_test = subset(reviews_numerical_test, select = -c(recommend_index))

rec_numerical_test = reviews_numerical$recommend_index[!train]

pred <- model.matrix(recommend_index ~ ., data=reviews_numerical)[, -1]
resp <- reviews_numerical$recommend_index
grid <- 10^seq(10, -2, length = 100) #potnetial values of lambda
train <- sample(1:nrow(pred), nrow(pred) / 2)
test <- (-train)
resp_test <- resp[test]

ridge_model <- glmnet(pred[train,], resp[train], alpha=0, lambda=grid)
cv <- cv.glmnet(pred[train,], resp[train], alpha=0)
best_lambda <- cv$lambda.min
best_lambda
min(cv$cvm)
plot(cv) #cv error as a function of lambda
predict(ridge_model, type = "coefficients", s = best_lambda)[1:27, ] #change to number of vars

lasso_model <- glmnet(pred[train,], resp[train], alpha=1, lambda=grid)
cv_lasso <- cv.glmnet(pred[train,], resp[train], alpha=1)
best_lambda_lasso <- cv_lasso$lambda.min
best_lambda_lasso
min(cv_lasso$cvm)
plot(cv_lasso) #cv error as a function of lambda
predict(lasso_model, type = "coefficients", s = best_lambda)[1:27, ] #change to number of vars
best_model <- glmnet(pred[train,], resp[train], alpha=1, lambda=best_lambda_lasso)
pred5 <- predict(best_model, s = best_lambda_lasso, newx = as.matrix(reviews_numerical_test))
print(pred5)
write(pred5, file = 'LASSO_output.txt')
mse5 <- mean((rec_numerical_test - pred5) ** 5)
mse5


#PCR model
#using the same model matrix as ridge and lasso
install.packages("pls")
library(pls)
pr.out <- prcomp(pred, scale = TRUE)
set.seed(1)
pcr.fit <- pcr(recommend_index ~ ., data = reviews_numerical, subset = train,scale = TRUE, validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")

pcr.pred <- predict(pcr.fit, pred[!train, ], ncomp = 5)
pcr.pred
mse6 <- mean((rec_test - pcr.pred)^2)
mse6

## Regression Tree
install.packages('tree')
library(tree)
attach(reviews_numerical)
tree_reviews <- tree(recommend_index ~ age + alike_feedback_count + division_name_Initmates + department_name_Dresses + department_name_Intimate + department_name_Jackets + department_name_Tops + department_name_Trend + class_name_Blouses + class_name_Dresses + class_name_Intimates + class_name_Jackets + class_name_Jeans + class_name_Knits + class_name_Layering + class_name_Legwear + class_name_Lounge + class_name_Outerwear + class_name_Pants + class_name_Shorts + class_name_Skirts + class_name_Sleep + class_name_Sweaters + class_name_Sweaters + class_name_Swim + class_name_Trend + rating_1 + rating_3 + rating_4 + rating_5, data = reviews_numerical)

plot(tree_reviews)
text(tree_reviews)
cv <- cv.tree(tree_reviews)
plot(cv$size, cv$dev)

yhat <- predict(tree_reviews, newdata = reviews_numerical_test)
yhat
mse7 <- mean((rec_numerical_test - yhat) ^ 2)
mse7

#I choose not to prune this tree as there are only 4 leaves in the current model


## Random Forests
install.packages('randomForest')
library(randomForest)
rf <- randomForest(recommend_index ~ ., data = reviews_numerical, subset = train, mtry = 6)
yhat_rf <- predict(rf, newdata = reviews_numerical_test)
yhat_rf
mse8 <- mean((rec_numerical_test - yhat_rf) ^ 2)
mse8
