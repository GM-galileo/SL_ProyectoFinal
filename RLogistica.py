import numpy as np

def ModeloRegresionLogistica_Predict(X,b,parametros):
    Matriz = np.asmatrix(np.column_stack((X,np.ones_like(X))))
    m,n = X.shape
    x_1 = np.ones([m, 1])
    x = np.column_stack([X,x_1])  

    logits = np.add(np.matmul(x,parametros), b)
    yhat = 1/(1 + np.exp(-logits)) 

    yhat = np.where(yhat>0.5,"Y","N")
    return yhat