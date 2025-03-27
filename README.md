# FCQ
Instructor Rating Trends and Clustering Project
Hi! I’m Ayush, a junior studying Data Science with a Business minor at CU Boulder. Welcome to my project repository where I analyze instructor ratings across University of Colorado campuses to uncover what distinguishes a good professor from a great one.

Project Overview
At CU, course evaluations are more than just scores—they reveal patterns in teaching effectiveness and student satisfaction that impact everything from classroom strategies to tenure decisions. In this project, I work with FCQ data from CU Boulder, Denver, Colorado Springs, and Anschutz to build predictive models (using SVM, Random Forest, Bagging, and Decision Trees) and apply clustering techniques (like K-Means, Random Forest proximity matrices, and Neural Newtworks through the use of autoencoders) to identify key predictors of instructor ratings.

Project Goals
Predictive Modeling: Forecast instructor ratings and compare model performance using RMSE, R-Squared, and MAE.

Feature Importance: Discover which variables most strongly influence student evaluations.

Clustering Analysis: Uncover natural groupings in the data and align these with categorical variables such as department and college.

Data Visualization: Create intuitive dashboards and plots to present findings to non-technical stakeholders.

Methodology
Data Preprocessing: Cleaned and reduced FCQ datasets from three CU campuses, including feature selection and necessary transformations.

Supervised Learning: Developed and cross-validated models in R to predict instructor ratings.

Unsupervised Learning: Applied dimensionality reduction and clustering techniques to extract hidden patterns.

Technical Tools: Utilized R packages such as readxl, MASS, caret, ggplot2, tidyverse, and more to execute and visualize the analyses.

Installation and Requirements
Ensure you have the following R packages installed:

r
Copy
install.packages(c("readxl", "MASS", "caret", "psych", "ggplot2", 
                   "caretEnsemble", "tidyverse", "mlbench", "flextable", 
                   "mltools", "tictoc", "ROSE", "ROCR", "rpart", "rpart.plot",
                   "randomForest", "cluster", "keras", "tensorflow"))
Running the Code
Data Loading: Data from Denver, Boulder, and Anschutz are loaded using read_excel.

Modeling: The repository includes scripts for supervised modeling (regression, classification) and unsupervised clustering.

Visualization: Results are visualized with ggplot2 and detailed dashboards are built for further insights.

Results & Future Work
This analysis provides a deeper understanding of the factors driving instructor ratings and highlights actionable insights for improving teaching effectiveness. Future work will focus on integrating more campuses, exploring temporal trends, and further refining our predictive and clustering models.

Thanks for checking out my project—feel free to reach out if you have any questions or suggestions!

Sincerely,
Ayush Uniyal
