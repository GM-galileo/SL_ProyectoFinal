import numpy as np

def Naive_Predict(X,m,v,p,y_class):
    Y_Pred = []
    for x in np.array(X):
        posteriors = []
        for idx, c in enumerate(y_class):
            prior = np.log(p[idx][0])

            numerator = np.exp(-((x - m[idx])**2)/(2 * v[idx]) )
            denominator = np.sqrt(2 * np.pi * v[idx])
            numl =np.log(numerator/denominator)

            posterior = np.sum(numl)
            posterior = prior + posterior

            posteriors.append(posterior)

        Y_Pred.append(y_class[np.argmax(posteriors)])
    return np.array(Y_Pred)