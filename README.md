# Single time-series prediction

I have deal with a predictive model whose task is to predict a future value based on historical data. Have to prepare the input and output pairs given the time series data.
I want to predict today's Dow Jones Index given available stock prices.You are aware of the RNN, or more precisely GRU network captures time-series patterns, we can build such a model with the input being the past three days' change values, and the output being the current day's change value. 
Gated recurrent unit (GRU) layers work using the same principle as LSTM, but they’re somewhat streamlined and thus cheaper to run (although they may not have as much representational power
as LSTM). This trade-off between computational expensiveness and representational power is seen everywhere in machine learning.
