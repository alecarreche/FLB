library(leaps)
library(caret)
library(ggplot2)

#### data import ####
data = read.csv("data/normalized_coaster_data.csv")
jax_data = read.csv("data/jax_data.csv")

#### model training ####

# set up cross validation
train.control <- trainControl(method = "cv", number = 10)

# fwd 
fwd <- train(success_metric ~., data = data[, -1],
                    method = "leapForward", 
                    tuneGrid = data.frame(nvmax = 1:35),
                    trControl = train.control
)
fwd$results
fwd$bestTune
sum = summary(fwd$finalModel)
models = list(fwd)

# svm regression linear kernel
svm_lin <- train(success_metric ~., data = data[, -1],
                    method = "svmLinear", 
                    trControl = train.control,
                   tuneLength=5,
                    scale=FALSE
)
models = append(models, list(svm_lin))

# svm regression radial kernel
svm_rad <- train(success_metric ~., data = data[, -1],
                   method = "svmRadial", 
                   trControl = train.control,
                   tuneLength=5, 
                   scale=FALSE
)
models = append(models, list(svm_rad))

# svm regression polynomial kernel 
svm_poly <- train(success_metric ~., data = data[, -1],
                       method = "svmPoly", 
                       trControl = train.control,
                       tuneLength=5, 
                       scale=FALSE
)
models = append(models, list(svm_poly))

#### residuals ####

predictions = data.frame(y=data[, "success_metric"])
for (model in models) {
  name = model$method
  predictions[ , name] = predict(model, data[, !names(data) %in% c("Coaster_Name", "success_metric")])
}

residuals = (predictions - predictions$y)[, -1]

index = c(1:nrow(residuals))
ggplot(residuals, aes(x=index, y=leapForward)) + 
  geom_point(aes(index, leapForward, col="Forward Stepwise")) + 
  geom_point(aes(index, svmLinear, col="Linear Kernel SVM")) + 
  geom_point(aes(index, svmRadial, col="Radial Kernel SVM")) +
  geom_point(aes(index, svmPoly, col="Polynomial Kernel SVM")) + 
  ggtitle("Residual Plot") + 
  xlab("Index") + 
  ylab("Residuals")

for (model in models) {
  results = model$results
  print(model$method)
  print(results[which.min(results$RMSE), "RMSE"])
}

#### jax predictions ####

jax_results = data.frame(jax_data$Coaster_Name)
jax_results$fwd = predict(fwd, jax_data)
jax_results$svm_lin = predict(svm_lin, jax_data)
jax_results$svm_rad = predict(svm_rad, jax_data)
jax_results$svm_poly = predict(svm_poly, jax_data)

#write.csv(jax_results, "data/results.csv")
