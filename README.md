# Project 5 - Social Recommendation Systems (CiaoDVD Dataset)
The aim of this project is to build a tool that implements and compares different types of recommendation algorithms on a real-world dataset. The technical aspects are described here below.

## Index
* [Dataset Description](#dataset)
* [Libraries and Modules](#libs)
* [Files Organization](#organization)
* [Memory Based Implementation](#meb)
* [Model Based Implementation](#mob)
* [Code Testing](#testing)

# <a name="dataset"></a> Dataset Description
The data used as source for this implementation can be found in https://www.librec.net/datasets.html **CiaoDVD**, which consist of 3 diferent *.txt files that contains information about some DVDs in 2013.

The particular file utilized in this case is [movie-ratings](Ciao-DVD-Datasets/movie-ratings.txt), which contains the following information:
1. File: movie-ratings.txt (size: 72,665 --> 72.7K)
2. Columns: userID, movieID, genreID, reviewID, movieRating, date

# <a name="libs"></a> Libraries and Modules
This implementation requires the instalation of the following tools:
* python version 3.xx
* pandas
* numpy
* matplotlib

The following tools are required only if __*.ipynb__ files are executed
* jupyterlab
* ipython

# <a name="organization"></a> Files Organization
```
📁 Ciao-DVD-Datasets
 └── movie-ratings.txt
📁 Images
📁 results
📁 support
 └── data_loading_analysis.py
 └── evaluation_metrics.py
 └── plots_custom.py
📄 main_pc.py
📄 main_pmf.py
📄 memory_based_cf.py
📄 model_based_cf.py
📓 Main_MeB.ipynb
📓 Main_MoB.ipynb
```
[main_pc.py](main_pc.py): Contains the main implementation for User based colaborative filtering RS using pearson correlation in pure python.

[Main_MeB.ipynb](Main_MeB.ipynb): The same code that main_pc.py  for jupyter notebook.

[main_pmf.py](main_pmf.py): Contains the main implementation for RS using Probabilistic Matrix Factorization in pure python.

[Main_MoB.ipynb](Main_MoB.ipynb): The same code that main_pmf.py for jupyter notebook.

[memory_based_cf.py](memory_based_cf.py): Contains all functions for User based colaborative filtering RS using pearson correlation.

* pearson_correlation
* neighborhood
* get_user_avgs
* predict

[model_based_cf.py](model_based_cf.py): Contains all functions for model of Probabilistic Matrix Factorization.

* train
* loss
* prediction

[support/data_loading_analysis.py](support/data_loading_analysis.py): dataset processing (loading, spliting, analysis data).

[support/evaluation_metrics.py](support/evaluation_metrics.py): RMSE and MAE functions.

[support/plots_custom.py](support/plots_custom.py): Customized functions to plot.

# <a name="meb"></a>Memory Based Implementation
### Details
Before to run the core of the algorithm there is an optional section to prune the dataset to reduce the size of the original dataset.
- The prune dataset method accepts parameters for prunning dataset either by user, movies and randomly.
- The data set is splited between 80% for training and 20% for testing.
- Calculate the Pearson correlation coeficient for the training dataset.

    ![Pearson Correlation](Images/pearson_corr.png)
- Neighborhood selection based on the previous person correaltion and K size value.
- Rating prediction calculation with weighted average rating

    ![Pearson Correlation](Images/prediction.png)
- Finally a loop for evaluation of the accuracy metrics Root Mean Squared Error - RMSE and Mean Absolute Error - MAE
- There is a section at the end of the code for writing tehe results of the test in external files, [results](results) folder

# <a name="mob"></a>Model Based Implementation
- For this implementation the dataset is splited in 60% for Trainig, 20% for Validation Data, 20% for Testing. This is made in the train_validate_test_split_pmf() function.
- Create number Rating matrix with shape (# of users, # movies), with training data.
- Unlike the previous implementation. PMF is based on a model, which is trained before to get the prediction, this recommender system works better against sparse matrices and face the scalability issues much better. 

    ![PMF](Images/Likelihood.png)
- The model's inputs are the following:
Parameters settings
    - lambda_u = 0.02
    - lambda_v = 0.02
    - learn_rate = 0.005
    - num_iters = 1000
    - latent_dim = from (5 to 50)
    - momentum = 0.9
- The objetive is obtain the latent user and movie feature matrices U, V after applying simple Stochastic Gradient Descent - SGD. 
    ![Loss_Function](Images/Loss_Function.png)
- Finally, we predict the prediction of the testing data using the model already trained
- Results stored in results/results_pmf1

# <a name="testing"></a> Code Testing
### Memory based: User based CF with Pearson Correlation

Before runing the code is required to set the parameters for prunning the dataset, although by default these parameters are set to run considering the full dataset values (this process can take huge time, mainly in the pearson correlation section). If you want to reduce the dataset size it is possible in 3 ways:
* Setting a min value for number of ratings by user. This is posible changing the variable _p_user_ to an int value and _how = 'u'_
* Setting a min value for number of ratings by movie. This is posible changing the variable _p_movie_ to an int value and _how = 'u'_
* Setting a number to prune randomly. This is posible seting variable _p_rnd_ with a number greater than 0 and less or equal than 1 and _how = 'r'_

The correlation matrix is computed for all users once, but if the training dataset changes, it is required re-compute the Pearson correlation matrix. This step is required because the target user should identify their most similar users. 

The testing by default run a loop to make multiple executions automatically, but the functions could be executed sequentially as follow to compute the neigborhood, and predictions just giving the k size in **get_neighborhood(k_size)** function.
```
get_neighborhood(k_size)
prediction, p_time = get_prediction()
rmse, mae = get_metrics(prediction)
```

### Model based: Probabilistic Matrix Factorization
