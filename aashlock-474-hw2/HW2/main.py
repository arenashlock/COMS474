import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

''' DATA FOR CALCULATIONS '''
X = np.array([1, 2, 3, 4, 5, 6, 10]).reshape(-1, 1)
Y = np.array([0, 1, 3, 2, 20, -6, 80])

homework_data = pd.read_csv("data/HW2.csv")
xs = homework_data["x"]
ys = homework_data["y"]
''' --------------------------------------------------- '''

''' GLOBAL CALCULATIONS '''
xm = np.mean(xs)
ym = np.mean(ys)
print("Sample mean (x): %.2f   (y): %.2f\n" % (xm, ym))
sxx = np.mean((xs-xm)**2)
syy = np.mean((ys-ym)**2)
print("Sample variance (x): %.2f   (y): %.2f " % (sxx, syy))
sxy = np.mean((xs-xm)*(ys-ym))
print("Sample covariance: %.2f\n" % sxy)
''' --------------------------------------------------- '''

''' y = ax + b CALCULATIONS '''
a = sxy/sxx
b = ym-(a*xm)
print("a: %.2f   b: %.2f\n" % (a, b))

RSS = np.sum(np.square(ys-(a*xs+b)))
print("RSS: %.2f" % RSS)
predictor1D = 1-(RSS/7)/syy
print("Predictor for 1D: %.2f\n" % predictor1D)
''' --------------------------------------------------- '''

''' y = ax + b MODEL '''
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(X, Y)
print("a = %s, b = %s, score = %s\n" %
      ("{:.2f}".format(model.coef_[0]),
       "{:.2f}".format(model.intercept_),
       "{:.2f}".format(model.score(X, Y))))
''' --------------------------------------------------- '''

''' y = kx CALCULATIONS '''
k = np.sum(xs*ys) / np.sum(xs**2)
print("k: %.2f\n" % k)

RSS = np.sum(np.square(ys-(k*xs)))
print("RSS: %.2f" % RSS)
predictor1F = 1-(RSS/7)/syy
print("Predictor for 1F: %.2f\n" % predictor1F)
''' --------------------------------------------------- '''

''' y = kx MODEL '''
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X, Y)
print("k = %s, score = %s\n" %
      ("{:.2f}".format(model.coef_[0]),
       "{:.2f}".format(model.score(X, Y))))
''' --------------------------------------------------- '''

''' y = ax + bx^2 + g CALCULATIONS '''
xmm = np.mean(xs**2)
sxX = np.sum((xs - xm)*(xs**2 - xmm))
sXX = np.sum(np.square(xs**2 - xmm))
sXy = np.sum((xs**2 - xmm)*(ys-ym))

alpha = ((sxy*sXX)-(sXy*sxX))/((sxx*sXX)-(sxX**2))
beta = (sXy*sxx-sxy*sxX)/(sxx*sXX-(sxX**2))
gamma = ym-alpha*xm-beta*xmm
print("alpha: %.2f   beta: %.2f   gamma: %.2f\n" % (alpha, beta, gamma))

RSS = np.sum(np.square(ys - (alpha * xs + beta * xs**2 + gamma)))
print("RSS: %.2f" % RSS)
predictor1H = 1 - (RSS/7)/syy
print("Predictor for 1H: %.2f\n" % predictor1H)
''' --------------------------------------------------- '''

''' y = ax + bx^2 + g MODEL '''
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(poly_features, Y)
print("alpha = %s, beta = %s, gamma = %s, score = %s" %
      ("{:.3f}".format(model.coef_[0]),
       "{:.3f}".format(model.coef_[1]),
       "{:.3f}".format(model.intercept_),
       "{:.3f}".format(model.score(poly_features, Y))))
''' --------------------------------------------------- '''