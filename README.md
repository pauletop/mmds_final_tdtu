# Mining Massive Datasets - Final Exam
## Brief Description

This repository contains the final exam of the Mining Massive Datasets course at the Ton Duc Thang University. The exam consists of 5 questions, each of which is a task related to a specific topic in data mining. The tasks are implemented using PySpark, a Python API for Apache Spark, to process large datasets in parallel and distributed systems. The exam covers the following topics:
## Requirements
Given the following datasets:
- `mnist_mini.csv`: a small subset of the MNIST dataset, containing 10000 rows and 785 columns. The first column is the label, and the remaining 784 columns are the pixel values of the grayscale 28x28 flattened images.
- `ratings2k.csv`: a dataset containing 2366 rows and 4 columns. The columns are `index`, `user`, `item`, and `rating` (0.0-5.0). The first row is the header and the remaining rows are the ratings given by users to items.
- `stocksHVN2022.csv`: a dataset containing 219 rows and 7 columns. The columns are `Ngay` (format: `dd/mm/yyyy`), `HVN` (closing price).
*for more details, please refer to the datasets in the [`datasets`](datasets) folder and the [`requirements_vni.pdf`] (requirements_vni.pdf) for the full requirements and [`finalreport_vni.pdf`] (finalreport_vni.pdf) for the report.*
## Question 1: Data Clustering
- Use the `mnist_mini.csv` dataset.
- Implement the K-means algorithm (from pyspark.ml.clustering.KMeans) with k=10. Where (các điểm dữ liệu tại dòng) data points at row index **0, 1, 2, 3, 4, 7, 8, 11, 18, 61** are weighted 10 times more than the other data points.
- For each cluster, report the average Euclidean distance from the centroid to each data point in the cluster. Visualize by using a bar chart.
## Question 2: Reducing Dimensionality
- Use the `mnist_mini.csv` dataset.
- Implement SVD (Singular Value Decomposition) to reduce the dimensionality of the dataset from 784 to 3.
- Randomly select 100 data points and visualize clustering results (from the previous question) in 3D space.
## Question 3: Collaborative Filtering
- Use the `ratings2k.csv` dataset.
- Implement the Alternating Least Squares (ALS) algorithm (from pyspark.ml.recommendation.ALS) to evaluate the model performance by using MSE with number of users "similar" in the range from 10 to 20.
- Run inference to illustrate the model's operation.
- Visualize MSE by using a bar chart.
## Question 4: Stock Prediction
- Use the `stocksHVN2022.csv` dataset.
- Predict fluctuations in the stock price of HVN (Vietnam Airlines) by using the Linear Regression (from pyspark.ml.regression.LinearRegression) algorithm and evaluate the model performance by using MSE. Visualize it by using a bar chart.
- The "fluctuation" is defined as:
    $amplitude = (HVN_{t} - HVN_{t-1}) / HVN_{t-1}$
    The first day is 0.0% by default.
- Features: 5-day previous closing prices.
- Target: the next day's (today's) closing price.
## Question 5: Multiclass Classification
- Use the `mnist_mini.csv` dataset.
- Implement the *Multi-layer Perceptron (MLP)*, *Random Forest*, and *Linear Support Vector Machine (SVM)* algorithms to classify the images.
   + *Input*: image vectors (784 features).
   + *Output*: labels (0-9).
   + *Loss function*: Cross Entropy.
   + *Evaluation metric*: Accuracy.

./.