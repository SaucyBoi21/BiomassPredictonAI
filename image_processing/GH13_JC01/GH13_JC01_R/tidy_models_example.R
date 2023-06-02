library(tidymodels)

# Set up the data and recipe
data("mtcars")
mtcars_train <- mtcars[1:24, ]
mtcars_test <- mtcars[25:32, ]
mtcars_recipe <- recipe(mpg ~ ., data = mtcars_train) %>% 
  step_normalize(all_predictors())

# Set up the model specification
rf_spec <- rand_forest(
  mtry = tune(),
  trees = tune()
) %>% 
  set_engine("ranger", importance = "impurity")

# Set up the resampling and tuning options
set.seed(123)
folds <- vfold_cv(mtcars_train, v = 5)
grid <- grid_latin_hypercube(
  mtry(range = c(4, 14)),
  trees(),
  sample = 10
)
tune_control <- control_grid(verbose = TRUE)

# Perform the hyperparameter tuning
rf_tune <- tune_grid(
  rf_spec,
  mtcars_recipe,
  resamples = folds,
  grid = grid,
  control = tune_control,
  metrics = metric_set(rmse),
  finalize_model = fit_model
)

# Print the results
rf_tune

