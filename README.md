# glucose-prediction

In this project, I predict the blood glucose concentration value in order to be able to generate an early warning if a hypoglycemic condition is going to occur.
The data simulates real-time sample acquisition via a CGM (Continuous Glucose Monitoring) sensor.

Prediction is performed using an __ordinary least squares Linear Regression model__.
For each new sample acquired, the model parameters (`m_tmp`, `q_tmp`) are re-estimated.
The user has the possibility to modify two parameters: `ph` and `mu`.
`ph` is the **prediction horizon** and specifies the time ( in minutes ) forward to estimate the blood glucose concentration. `mu` is the forgetting factor. The mu^k coefficient represents the weight of the blood glucose sample at k istants before the current sampling time. Being 0<mu≤1, the newer sample has greater weight; the higher the mu, the longer the memory, the lower the mu, the faster you forget about past data.

An example of a prediction with `ph = 30` and `mu = 0.9` is shown in the figure:

![alt text](https://github.com/lorrandal/glucose-prediction/blob/master/prediction.svg)

Note that the predicted signal crosses the hypoglycemic threshold twice. In the first case in t ≈ 730 minutes this is a false positive. In the second case in t ≈ 840 minutes, the model correctly predicts the event about 20 minutes in advance.

The data is contained in the `data` folder. `ys.csv` contains blood sugar values, `ts.csv` is the time vector in which the values of `ys` were sampled.

The dataset was provided in the class of Analysis of biological data, held by Professor Sparacino, Bioengineering, Department of Information Engineering, University of Padua, academic year 2015/2016.
