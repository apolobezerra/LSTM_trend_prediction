# LSTM_trend_prediction


This project is an in progress code with the intent of predicting daily stock price trends using LSTM models.
It is an adapted version of the code in https://github.com/nayash/Stock-Price-Prediction/blob/master/stock_pred_main.py.
Thus, it still has some problems to be fixed, such as non-lagged variables and non-sufficient features for a good finance model.

It also needs a more appropriate forecast KPI. The MSE isn't the best KPI for trend pedictions. Accuracy is a best evaluator for this case.

Further improvements can be done by denoising the data set with wavelet transformations, adding features of technical and sentiment analysis and ensembling Deep Learning methods

Some good articles on the subject:

Z. Zhao, R. Rao, S. Tu and J. Shi, "Time-Weighted LSTM Model with Redefined Labeling for Stock Trend Prediction", 
2017 IEEE 29th International Conference on Tools with Artificial Intelligence (ICTAI), Boston, MA, 2017, pp. 1210-1217, 
doi: 10.1109/ICTAI.2017.00184.

Jiayu Qiu,Bin Wang ,Changjun Zhou, "Forecasting stock prices with long-short term memory neural network based on attention mechanism",
https://doi.org/10.1371/journal.pone.0227222
