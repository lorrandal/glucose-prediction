import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#%% FIRST MODEL FIT WITH THE WHOLE DATA SET
# IMPORT DATA

ys = np.genfromtxt(fname='data/ys.csv', delimiter=',')
ts = np.genfromtxt(fname='data/ts.csv', delimiter=',')

#converting numpy array to pandas DataFrame
ys = pd.DataFrame(ys)
ts = pd.DataFrame(ts)
   
#%% ACTUAL PREDICTION, WITH WEIGHTS
N = len(ys)
PH = 20 #prediction horizon
mu = 0.9

tp_tot = np.zeros(N)
yp_tot = np.zeros(N)


for i in range(1, N+1):
    ts_tmp = ts[0:i]
    ys_tmp = ys[0:i]
    ns = len(ys_tmp)
    
    weights = np.ones(ns)*mu
    for k in range(ns):
        weights[k] = weights[k]**k
    weights = np.flip(weights, 0)
        

    # MODEL
    lm_tmp = linear_model.LinearRegression()    
    model_tmp = lm_tmp.fit(ts_tmp, ys_tmp, sample_weight=weights)
    m_tmp = model_tmp.coef_
    q_tmp = model_tmp.intercept_

    # PREDICTION
    tp = ts[0][ns-1] + PH
    yp = m_tmp*tp + q_tmp
    
    
    tp_tot[i-1] = tp    
    yp_tot[i-1] = yp



#fig = plt.figure()
fig, ax = plt.subplots()
fig.suptitle('Running line forecasting', fontsize=14, fontweight='bold')
#ax = fig.add_subplot(111)
ax.set_title('mu = %g, PH=%g ' %(mu, PH))
ax.plot(tp_tot, yp_tot, '--', label='prediction') 
ax.plot(ts, ys, label='original') 
ax.set_xlabel('time [min]')
ax.set_ylabel('glucose (mg/dl)')
ax.legend()
    
    

    