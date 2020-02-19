# Predicting street dust (PM10) concentrations with dilated causal CNN
A model for predicting street dust concentrations is developed in this project. Model is implemented with PyTorch. Model architecture contains stacks of dilated convolutions, residual connections, a SELU activation function and conditioning with multivariate input data. Input data is hourly observations of PM10 concentrations, amount of rain, temperature and wind speed of 28 months. Input data divided into training, validation and test sets with moving window method, and scaled to zero mean and unit variance before feeding to model. Model outputs directly the 48 hours forecast, and the same model is used for making the predictions for all the test batches. Model architecture has got inspiration from adapted WaveNet network by Borovykh et al., 2018. Model is run with GPU on Colab.

This is a project work for the Deep Learning course in Aalto University. 

A. Borovykh, S. Bohte, and C. W. Oosterlee, Conditional Time Series Forecasting with Convolutional Neural Networks, ArXiv e-prints, (2018)
