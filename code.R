library(tidyverse)
library(lubridate)
library(caret)

#read files
edx <- as.data.frame(read_csv('data/edx.csv'))
validation <- as.data.frame(read_csv('data/validation.csv'))

#make an index for genres
gen_ind <- edx %>% group_by(genres) %>% summarise(genres=genres[1]) %>% mutate(gen_ind=seq(1:n()))
edx <- edx %>% left_join(gen_ind, by = "genres")
validation <- validation %>% left_join(gen_ind, by = "genres")

#make columns with movie release and rating year
edx <- edx %>% mutate(year = as.numeric(str_extract(str_extract(title, "[(]\\d{4}[)]"), "\\d{4}")), r_date = year(as_datetime(timestamp)))
validation <- validation %>% mutate(year = as.numeric(str_extract(str_extract(title, "[(]\\d{4}[)]"), "\\d{4}")), r_date = year(as_datetime(timestamp)))

#create train and test set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")
#save(train_set, file="rdas/train-set.rda")
#save(test_set, file="rdas/test-set.rda")

#loss function
RMSE <- function(true_ratings, predicted_ratings){sqrt(mean((true_ratings - predicted_ratings)^2))}

#effect function with regularization
EFFECT <- function(d, l){
  #movie effect
  b_i <- d %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  train_set <- d %>% left_join(b_i, by = "movieId")
  #user effect
  b_u <- train_set %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  train_set <- train_set %>%  left_join(b_u, by = "userId")
  #genres effect
  b_g <- train_set %>% 
    group_by(gen_ind) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  train_set <- train_set %>%  left_join(b_g, by = "gen_ind")
  #release year effect with smooth function
  b_y <- train_set %>% 
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
  fit_y <- loess(b_y ~ year, degree=1, span = 0.05, b_y, family="symmetric")
  #save(fit_y, file="rdas/fit-y.rda")
  b_y_hat <- predict(fit_y, newdata = b_y)
  b_y <- b_y %>% mutate(b_y = b_y_hat)
  train_set <- train_set %>%  left_join(b_y, by = "year")
  #rate year effect with smooth function
  b_r <- train_set %>% 
    group_by(r_date) %>%
    summarize(b_r = sum(rating - b_i - b_u - b_g - b_y - mu)/(n()+l))
  fit_r <- loess(b_r ~ r_date, degree=1, span = 0.3, b_r, family="symmetric")
  #save(fit_r, file="rdas/fit-r.rda")
  b_r_hat <- predict(fit_r, newdata = b_r)
  b_r <- b_r %>% mutate(b_r = b_r_hat)
  train_set <- train_set %>%  left_join(b_r, by = "r_date")
  new_train <- train_set
  train_set <- train_set %>% select(-b_i, -b_u, -b_g, -b_y, -b_r)
  new_train}

#set best lambda
lambdas <- seq(4, 6, 0.1)
mu <- mean(train_set$rating)
rmses <- sapply(lambdas, function(l){
  new_train <- EFFECT(train_set, l)
  b_i <- new_train %>% group_by(movieId) %>% summarise(b_i = b_i[1])
  b_u <- new_train %>% group_by(userId) %>% summarise(b_u = b_u[1])
  b_g <- new_train %>% group_by(gen_ind) %>% summarise(b_g = b_g[1])
  b_y <- new_train %>% group_by(year) %>% summarise(b_y = b_y[1])
  b_r <- new_train %>% group_by(r_date) %>% summarise(b_r = b_r[1])
  #make prediction
  pred <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "gen_ind") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_r, by = "r_date")
  pred <- pred %>% mutate(y_hat = mu + ifelse(is.na(b_i), 0, b_i) + ifelse(is.na(b_u), 0, b_u) 
                          + ifelse(is.na(b_y), 0, b_y) + ifelse(is.na(b_g), 0, b_g) + ifelse(is.na(b_r), 0, b_r))
  RMSE(test_set$rating, pred$y_hat)
})
best_lambda <- data.frame(lambdas = lambdas, rmses = rmses)
#save(best_lambda, file="rdas/best-lambda.rda")

# make a new train set with best lambda and biases
l <- lambdas[which.min(rmses)]
new_train <- EFFECT(train_set, l)

#add residual column
new_train <- new_train %>% mutate(resid = rating - mu - b_i - b_u - b_y - b_g - b_r)
b_i <- new_train %>% group_by(movieId) %>% summarise(b_i = b_i[1])
b_u <- new_train %>% group_by(userId) %>% summarise(b_u = b_u[1])
b_g <- new_train %>% group_by(gen_ind) %>% summarise(b_g = b_g[1])
b_y <- new_train %>% group_by(year) %>% summarise(b_y = b_y[1])
b_r <- new_train %>% group_by(r_date) %>% summarise(b_r = b_r[1])
#save(new_train, file="rdas/new-train.rda")

#PRED function
PRED <- function(d){
  pred <- d %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "gen_ind") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_r, by = "r_date")
  pred <- pred %>% mutate(y_hat = mu + ifelse(is.na(b_i), 0, b_i) 
    + ifelse(is.na(b_u), 0, b_u) + ifelse(is.na(b_y), 0, b_y) 
    + ifelse(is.na(b_g), 0, b_g) + ifelse(is.na(b_r), 0, b_r))  %>% 
    mutate(y_hat = ifelse(y_hat <= 0.5, 0.5, ifelse(y_hat >= 5, 5, y_hat)))
  pred
}

#make prediction
pred <- PRED(test_set)

#discards users and movies with few ratings
n_movies <- new_train %>% group_by(movieId) %>% summarise(N = n())
m_n_movies <- mean(n_movies$N)
n_users <- new_train %>% group_by(userId) %>% summarise(N = n())
m_n_users <- mean(n_users$N)
new_train_small <- new_train %>% 
  group_by(movieId) %>%
  filter(n() >= m_n_movies) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= m_n_users) %>% ungroup()

#residual predictions by linear regression
fit_lm <- train(resid ~ movieId + userId + gen_ind + year + r_date, method = "lm", new_train_small)
#save(fit_lm, file="rdas/fit-lm.rda")
resid_lm <- predict(fit_lm, pred)
pred$resid_lm <- resid_lm
pred <- pred %>% mutate(pred_lm = ifelse(y_hat+resid_lm <= 0.5, 0.5, ifelse(y_hat + resid_lm >= 5, 5, y_hat+resid_lm)))

#residual predictions by logistc regression
fit_glm <- train(resid ~ movieId + userId + gen_ind + year + r_date, method = "glm", new_train_small)
#save(fit_glm, file="rdas/fit-glm.rda")
resid_glm <- predict(fit_glm, pred)
pred$resid_glm <- resid_glm
pred <- pred %>% mutate(pred_glm = ifelse(y_hat+resid_glm <= 0.5, 0.5, ifelse(y_hat+resid_glm >= 5, 5, y_hat+resid_glm)))

#residual predictions by local weighted regression
fit_loess <- train(resid ~ movieId + userId + gen_ind + year + r_date, method = "gamLoess", new_train_small)
#save(fit_loess, file="rdas/fit-loess.rda")
resid_loess <- predict(fit_loess, pred)
pred$resid_loess <- resid_loess
pred <- pred %>% mutate(pred_loess = ifelse(y_hat+resid_loess <= 0.5, 0.5, ifelse(y_hat+resid_loess >= 5, 5, y_hat+resid_loess)))

#ensemble
pred <- pred %>% mutate(ensemble = (pred_lm + pred_glm + pred_loess)/3)
pred <- pred %>% mutate(ensemble = ifelse(ensemble <= 0.5, 0.5, ifelse(ensemble >= 5, 5, ensemble)))
#save(pred, file="rdas/pred.rda")

#effect function with edx
new_edx <- EFFECT(edx, l)

#add resid column
new_edx <- new_edx %>% mutate(resid = rating - mu - b_i - b_u - b_y - b_g - b_r)
b_i <- new_edx %>% group_by(movieId) %>% summarise(b_i = b_i[1])
b_u <- new_edx %>% group_by(userId) %>% summarise(b_u = b_u[1])
b_g <- new_edx %>% group_by(gen_ind) %>% summarise(b_g = b_g[1])
b_y <- new_edx %>% group_by(year) %>% summarise(b_y = b_y[1])
b_r <- new_edx %>% group_by(r_date) %>% summarise(b_r = b_r[1])

#make prediction
valid <- PRED(validation)

#discards users and movies with few ratings
n_movies <- new_edx %>% group_by(movieId) %>% summarise(N = n())
m_n_movies <- mean(n_movies$N)
n_users <- new_edx %>% group_by(userId) %>% summarise(N = n())
m_n_users <- mean(n_users$N)
new_edx_small <- new_edx %>% 
  group_by(movieId) %>%
  filter(n() >= m_n_movies) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= m_n_users) %>% ungroup()

#residual predictions by linear regression
fit_lm <- train(resid ~ movieId + userId + gen_ind + year + r_date, method = "lm", new_edx_small)
resid_lm <- predict(fit_lm, valid)
valid$resid_lm <- resid_lm
valid <- valid %>% mutate(pred_lm = ifelse(y_hat+resid_lm <= 0.5, 0.5, ifelse(y_hat + resid_lm >= 5, 5, y_hat+resid_lm)))

#residual predictions by logistc regression
fit_glm <- train(resid ~ movieId + userId + gen_ind + year + r_date, method = "glm", new_edx_small)
resid_glm <- predict(fit_glm, valid)
valid$resid_glm <- resid_glm
valid <- valid %>% mutate(pred_glm = ifelse(y_hat+resid_glm <= 0.5, 0.5, ifelse(y_hat+resid_glm >= 5, 5, y_hat+resid_glm)))

#residual predictions by local weighted regression
fit_loess <- train(resid ~ movieId + userId + gen_ind + year + r_date, method = "gamLoess", new_edx_small)
resid_loess <- predict(fit_loess, valid)
valid$resid_loess <- resid_loess
valid <- valid %>% mutate(pred_loess = ifelse(y_hat+resid_loess <= 0.5, 0.5, ifelse(y_hat+resid_loess >= 5, 5, y_hat+resid_loess)))

#ensemble
valid <- valid %>% mutate(ensemble = (pred_lm + pred_glm + pred_loess)/3)
valid <- valid %>% mutate(ensemble = ifelse(ensemble <= 0.5, 0.5, ifelse(ensemble >= 5, 5, ensemble)))
save(valid, file="rdas/valid.rda")

#RMSE
RMSE(validation$rating, valid$ensemble)
