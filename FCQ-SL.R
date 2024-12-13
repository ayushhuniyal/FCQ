install.packages("readxl")
library(readxl)
library(MASS)
library(caret) #for training machine learning models
library(psych) ##for description of  data
library(ggplot2) ##for data visualization
library(caretEnsemble)##enables the creation of ensemble models
library(tidyverse) ##for data manipulation
library(mlbench)  ## for benchmarking ML Models
library(flextable) ## to create and style tables
library(mltools) #for hyperparameter tuning
library(tictoc) #for determining the time taken for a model to run
library(ROSE)  ## for random oversampling
#library(smotefamily) ## for smote sampling
library(ROCR) ##For ROC curve
library(rpart) #### For decision trees
library(rpart.plot)###For plotting decision trees


#Load data 1
data_denver <- read_excel("/Users/ayushuniyal/Downloads/denver.xlsx", sheet = "Data", skip = 6)
head(data_denver)

#Load data 2
data_boulder <- read_excel("/Users/ayushuniyal/Downloads/boulder.xlsx", sheet = "Data", skip = 6)
head(data_boulder)

#Load data 3
data_anschutz <- read_excel("/Users/ayushuniyal/Downloads/anschutz.xlsx", sheet = "Data", skip = 6)
head(data_anschutz)

names(data_anschutz)
names(data_boulder)
names(data_denver)

#reducing data size by choosing 10% of the data randomly
set.seed(202)
denver_clean <- data_denver %>% na.omit() %>% sample_frac(size=0.05, replace = FALSE) %>% select(-Sbjct, -Crse, -Sect, -'Crse Title', -Instructor)#, -Term, -College, -Dept, -'Crse Lvl')

#ensure no missing entries
sum(is.na(denver_clean))

#training and testing split
set.seed(321)
rows <- sample(x = 0.7*nrow(denver_clean), replace = FALSE)

training <- denver_clean[rows,]
testing <- denver_clean[-rows,]
dim(training)
dim(testing)



#lin_reg <- lm(Instr~., data = training[1:1000,])
#summary(lin_reg)

#feature selection through step wise regression 
set.seed(304)
lin_reg_reduced <- stepAIC(lm(Instr ~ ., data = training), direction = "both")
summary(lin_reg_reduced)

#Goals
#Feedback
#Engaged
#Caring
#Crse Effect
#Crse Learned
#CritThnk 

#Cross validation control
control1 = trainControl(method='cv', number=10)


#Support Vector Machine

# Set the seed for reproducibility
set.seed(123)

# Start timing the training process
tic()

# Train the SVM regression model using a radial kernel
SvmModel <- train(Instr ~ Goals+Feedback+Engaged+Caring+CrseEffect+CrseLearned+CritThnk, 
                  data = training, 
                  method = "svmRadial", 
                  trControl = control1)

# Stop timing
toc()

# Output the model summary
SvmModel

# Evaluate the model on the test set
Svmpred <- predict(SvmModel, newdata = testing)

# Calculate model evaluation metrics (e.g., RMSE, MAE)
SVM.metrics <- postResample(Svmpred, testing$Instr)

# Output metrics
SVM.metrics

# Plot variable importance
#plot(varImp(SvmModel, scale = TRUE), main = "SVM Variable Importance")
#doesn't seem to work. 


#Random Forest

set.seed(123)

tic()

RFModel <- train(Instr ~ Goals+Feedback+Engaged+Caring+CrseEffect+CrseLearned+CritThnk, data=training, method="rf", trControl=control1, tuneLength=5)

toc()

RFModel

RFpred<-predict(RFModel,newdata = testing)

RF.metrics <- postResample(RFpred, testing$Instr)

RF.metrics

plot(varImp(RFModel, scale=T))



#Bagging

set.seed(123)

tic()

bagModel <- train(Instr ~ Goals+Feedback+Engaged+Caring+CrseEffect+CrseLearned+CritThnk, data=training, method="treebag", trControl=control1)

toc()

bagModel

bagpred<-predict(bagModel,newdata = testing)

bag.metrics <- postResample(bagpred, testing$Instr)

bag.metrics

plot(varImp(bagModel, scale=T))


#Decision Trees

set.seed(123)

tic()

DTModel <- train(Instr ~ Goals+Feedback+Engaged+Caring+CrseEffect+CrseLearned+CritThnk, data=training, method="rpart",preProc=c("center", "scale"), trControl=control1)

toc()

DTModel

DTpred<-predict(DTModel,newdata = testing)

DT.metrics <- postResample(DTpred, testing$Instr)

DT.metrics

plot(varImp(DTModel, scale=T))

rpart.plot(DTModel$finalModel)

suppressMessages(library(rattle))

fancyRpartPlot(DTModel$finalModel)







##################Unsupervised Modeling################




set.seed(202)
boulder_clean <- data_boulder %>% sample_frac(size = 0.025, replace = FALSE) %>% na.omit() %>% select(-'Instructor Name',-'Crse Title', -Term, -'Instr Grp') #dataset with 1430 observations
head(boulder_clean)


###Preprocessing the 'DEPT' variable
unique_depts <- unique(boulder_clean$Dept)
print(unique_depts) #what are all the unique departments

print(table(boulder_clean$Dept)) #a count of all unique departments

library(forcats)

# Collapse departments appearing fewer than a threshold in this case top 10
boulder_clean$Dept <- fct_lump_min(boulder_clean$Dept, min = 15)

# Inspect the updated levels
levels(boulder_clean$Dept)

##Picking which variables to use for clustering process


summary(boulder_clean)


#visualize distributions
library(ggplot2)
ggplot(boulder_clean, aes(x = Crse)) + geom_histogram(binwidth = 0.5) + theme_minimal()

ggplot(boulder_clean, aes(x = Feedback)) + geom_histogram(binwidth = 0.5) + theme_minimal()

ggplot(boulder_clean, aes(x = Interact)) + geom_histogram(binwidth = 0.5) + theme_minimal()

ggplot(boulder_clean, aes(x = Reflect)) + geom_histogram(binwidth = 0.5) + theme_minimal()

ggplot(boulder_clean, aes(x = Eval)) + geom_histogram(binwidth = 0.5) + theme_minimal()

ggplot(boulder_clean, aes(x = Collab)) + geom_histogram(binwidth = 0.5) + theme_minimal()

ggplot(boulder_clean, aes(x = Contrib)) + geom_histogram(binwidth = 0.5) + theme_minimal()

ggplot(boulder_clean, aes(x = Enroll)) + 
  geom_histogram(binwidth = 10, fill = "green", alpha = 0.7) + 
  theme_minimal()

#ggplot(boulder_clean, aes(x = 'Resp Rate')) + 
  #geom_histogram(binwidth = 10, fill = "green", alpha = 0.7) + 
  #theme_minimal()

#calculate skewness to see if transformations are necessary
library(e1071)
skewness(boulder_clean$Feedback) #sqrt transformation
skewness(boulder_clean$Enroll) #log transformation
skewness(boulder_clean$Interact) #sqrt transformation
skewness(boulder_clean$Reflect) #sqrt transformation
skewness(boulder_clean$Collab) #sqrt transformation
skewness(boulder_clean$Contrib)
skewness(boulder_clean$Eval)
skewness(boulder_clean$`Resp Rate`) #no trans needed

#Transformations
boulder_clean$Enroll <- log1p(boulder_clean$Enroll) 
boulder_clean$Interact <- sqrt(max(boulder_clean$Interact) - boulder_clean$Interact)

ggplot(boulder_clean, aes(x = Interact)) + geom_histogram(binwidth = 0.5) + theme_minimal()

#For Negatively Skewed Data:
#Negative skew means most of the data is concentrated near higher values, with a long tail of smaller values.
#A reverse square root flips the data so that the square root transformation can then appropriately compress the higher concentration of
#values near the tail.

#Scaling variables
clustering_vars <- boulder_clean[, c("Feedback", "Reflect", "Collab", "Contrib", 
                                     "Enroll", "Resp Rate", "Interact")]

# Scale the selected variables
boulder_scaled <- scale(clustering_vars)

# Check the scaled data
summary(boulder_scaled)


####K-means clustering####

# Set a seed for reproducibility
set.seed(123)

# Start timing
tic()

# Select features for clustering
clustering_data <- boulder_clean %>%
  select(Feedback, Reflect, Collab, Contrib, Enroll, `Resp Rate`, Interact) %>%
  na.omit()  # Ensure no missing values

# Apply K-Means clustering with 2 clusters
kmeans_model <- kmeans(clustering_data, centers = 2, nstart = 25)

# Stop timing
toc()

# Print the clustering results
print(kmeans_model)

# Evaluate the clustering with the silhouette score
library(cluster)
silhouette_scores <- silhouette(kmeans_model$cluster, dist(clustering_data))
mean_silhouette <- mean(silhouette_scores[, 3])
cat("Mean Silhouette Score:", mean_silhouette, "\n")

# Visualize clusters (PCA for dimensionality reduction)
library(ggplot2)
pca <- prcomp(clustering_data, center = TRUE, scale. = TRUE)
pca_data <- as.data.frame(pca$x[, 1:2])  # First two principal components
pca_data$Cluster <- as.factor(kmeans_model$cluster)

ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 2) +
  labs(title = "K-Means Clustering Results (PCA Projection)") +
  theme_minimal()


#calculating optimal number of clusters which is 2
library(cluster)
silhouette_scores <- sapply(2:10, function(k) {
  km <- kmeans(clustering_data, centers = k, nstart = 25)
  silhouette_score <- mean(silhouette(km$cluster, dist(clustering_data))[, 3])
  return(silhouette_score)
})
plot(2:10, silhouette_scores, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters (k)", ylab = "Mean Silhouette Score")

#adding the cluster assignments to dataset
boulder_clean$Cluster <- as.factor(kmeans_model$cluster)

# Count the number of instances of each Dept within each cluster
dept_cluster_distribution <- boulder_clean %>%
  group_by(Cluster, Dept) %>%
  summarize(count = n(), .groups = "drop")

# View the distribution
print(dept_cluster_distribution)


ggplot(boulder_clean, aes(x = College, fill = Cluster)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Cluster Distribution Across Colleges", x = "College", y = "Proportion")

#converting to percentages 
cluster_summary <- dept_cluster_distribution %>%
  group_by(Cluster) %>%
  mutate(Percentage = count / sum(count) * 100) %>%
  arrange(Cluster, desc(Percentage))

print(cluster_summary)

#Department stacked barplot in relation with clusters
ggplot(cluster_summary, aes(x = Cluster, y = Percentage, fill = Dept)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Proportion of Departments Within Each Cluster", x = "Cluster", y = "Percentage")










######Random Forest Proximity Matrices for Clustering######
# Set a seed for reproducibility
set.seed(123)

# Start timing
tic()

# Train a Random Forest model without a response variable
library(randomForest)

rf_model <- randomForest(
  x = boulder_clean %>%
    select(Feedback, Reflect, Collab, Contrib, Enroll, `Resp Rate`, Interact) %>%
    na.omit(),  # Ensure no missing values
  ntree = 500,       # Number of trees in the forest
  proximity = TRUE,  # Compute proximity matrix
  oob.prox = TRUE    # Use out-of-bag samples for proximity
)

# Stop timing
toc()

# Extract the proximity matrix
proximity_matrix <- rf_model$proximity

# Apply hierarchical clustering to the proximity matrix
hc <- hclust(as.dist(1 - proximity_matrix), method = "ward.D2")

# Cut the dendrogram into 2 clusters (based on optimal clusters found earlier)
rf_clusters <- cutree(hc, k = 2)

# Visualize the dendrogram
plot(hc, main = "Random Forest Proximity-Based Clustering", cex = 0.6)
rect.hclust(hc, k = 2, border = "red")  # Highlight clusters

# Evaluate clustering with silhouette score
library(cluster)
silhouette_scores <- silhouette(rf_clusters, as.dist(1 - proximity_matrix))
mean_silhouette <- mean(silhouette_scores[, 3])
cat("Mean Silhouette Score:", mean_silhouette, "\n")

# Add the Random Forest clusters to the dataset
boulder_clean$RF_Cluster <- as.factor(rf_clusters)

# Analyze cluster distribution by Department
rf_cluster_summary <- boulder_clean %>%
  group_by(RF_Cluster, Dept) %>%
  summarize(count = n(), .groups = "drop") %>%
  mutate(Percentage = count / sum(count) * 100)

print(rf_cluster_summary)

# Visualize the cluster distribution across Departments
ggplot(rf_cluster_summary, aes(x = RF_Cluster, y = Percentage, fill = Dept)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Proportion of Departments Within Each RF Cluster", x = "Cluster", y = "Percentage")


ggplot(rf_cluster_summary, aes(x = RF_Cluster, y = Percentage, fill = College)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Proportion of Departments Within Each RF Cluster", x = "Cluster", y = "Percentage")



######Neural-Networks with the use of an autoencoder######

library(keras)
library(tensorflow)
library(tidyverse)

# Preprocess data
boulder_clean <- boulder_clean %>%
  select(Feedback, Reflect, Collab, Contrib, Enroll, `Resp Rate`, Interact) %>%
  na.omit()  # Remove missing values

# Normalize the data (Min-Max Scaling)
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
boulder_clean <- as.data.frame(lapply(boulder_clean, normalize))

# Convert to matrix for Keras
data_matrix <- as.matrix(boulder_clean)

# Define Autoencoder Architecture
input_dim <- ncol(data_matrix)  # Number of input features

# Encoder
encoder <- keras_model_sequential() %>%
  layer_dense(units = 7, activation = 'relu', input_shape = c(input_dim)) %>%  # Hidden layer
  layer_dense(units = 3, activation = 'relu')  # Latent representation layer

# Decoder
decoder <- keras_model_sequential() %>%
  layer_dense(units = 7, activation = 'relu', input_shape = c(3)) %>%  # Hidden layer
  layer_dense(units = input_dim, activation = 'sigmoid')  # Output layer

# Combine Encoder and Decoder into an Autoencoder
autoencoder <- keras_model(inputs = encoder$input, outputs = decoder(encoder$output))

# Compile the Autoencoder
autoencoder %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(learning_rate = 0.01)
)

# Train the Autoencoder
history <- autoencoder %>% fit(
  x = data_matrix,
  y = data_matrix,
  epochs = 100,  # Train for more epochs for better reconstruction
  batch_size = 32,
  validation_split = 0.2
)

# Plot the training history
plot(history)

# Extract Latent Features
encoder_model <- keras_model(inputs = encoder$input, outputs = encoder$output)
latent_features <- encoder_model %>% predict(data_matrix)

# Perform K-Means Clustering on Latent Features
set.seed(123)
kmeans_model <- kmeans(latent_features, centers = 2)  # Optimal number of clusters determined earlier

# Add Clustering Results to the Data
boulder_clean$Cluster <- as.factor(kmeans_model$cluster)

# Visualize Clusters Using PCA on Latent Features
pca_latent <- prcomp(latent_features)
pca_data <- as.data.frame(pca_latent$x[, 1:2])  # First two principal components
pca_data$Cluster <- boulder_clean$Cluster

ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 2) +
  labs(title = "Clusters Based on Autoencoder Latent Features", x = "PC1", y = "PC2") +
  theme_minimal()

# Evaluate Clustering Quality with Silhouette Score
library(cluster)
silhouette_scores <- silhouette(kmeans_model$cluster, dist(latent_features))
mean_silhouette <- mean(silhouette_scores[, 3])
cat("Mean Silhouette Score:", mean_silhouette, "\n")

# Analyze Cluster Alignment with Categories (e.g., Dept)
cluster_summary <- boulder_clean %>%
  group_by(Cluster) %>%
  summarize(across(everything(), mean), .groups = "drop")

print(cluster_summary)









