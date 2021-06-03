# Básicamente estamos tratando de reconstruir las funciones y clases que ya 
# vienen implementadas en Python dentro de la librería scikit learn:
#   - sklearn.linear_model.Ridge
#   - sklearn.linear_model.Lasso
#   - sklearn.metrics.mean_squared_error
#   - sklearn.model_selection.cross_validate
#   - sklearn.linear_model.RidgeCV
#   - sklearn.linear_model.LassoCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Grafica la matriz de correlación de las variables explicativas.
def correlation_matrix(X):
    plt.figure(figsize = (8, 7))
    plt.title('Matriz de correlación')
    plt.xticks(np.arange(X.shape[1]), X.columns)
    plt.yticks(np.arange(X.shape[1]), X.columns)
    plt.imshow(np.abs(np.corrcoef(X, rowvar = False)), cmap = 'Blues')
    plt.colorbar()
    plt.show()

# Regresión Ridge.
class Ridge:    
    # Constructor.
    def __init__(self, alpha, fit_intercept = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
    
    # Estima los parámetros.
    def fit(self, X, y):
        n, p = X.shape
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if self.fit_intercept:
            X = np.column_stack((np.ones(n), X))
            p += 1

        self.coef = np.linalg.solve(X.T @ X + self.alpha*np.eye(p), X.T @ y)
        if self.fit_intercept:
            self.intercept = self.coef[0]
            self.coef = self.coef[1:]
        else:
            self.intercept = 0

        return self
        
    # Predicción.
    def predict(self, X):
        return self.intercept + X.dot(self.coef)
    
    # Coeficiente de determinación.
    def score(self, X, y):
        y_pred = self.predict(X)
        residual_sum_of_squares = np.sum((y - y_pred)**2)
        total_sum_of_squares = np.sum((y - np.mean(y))**2)
        return 1 - residual_sum_of_squares / total_sum_of_squares

# Regresión LASSO
class Lasso:    
    # Constructor.
    def __init__(self, alpha, fit_intercept = True, max_iter = 10000, tol = 
                 1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
    
    # Estima los parámetros usando descenso por coordenadas.
    def fit(self, X, y):
        n, p = X.shape
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if self.fit_intercept:
            X = np.column_stack((np.ones(n), X))
            p += 1
        
        self.coef = np.linalg.solve(X.T @ X + self.alpha*np.eye(p), X.T @ y)
        for self.n_iter in range(1, self.max_iter + 1):
            coef_old = self.coef.copy()
            for i in range(p):
                X_i_deleted = np.delete(X, i, axis = 1)
                coef_i_deleted = np.delete(self.coef, i)       
                X_i = X[:, i]
                num = np.dot(y - X_i_deleted.dot(coef_i_deleted), X_i)
                den = X_i.dot(X_i)
                if num > self.alpha/2:
                    self.coef[i] = (num - self.alpha/2) / den
                elif num < -self.alpha/2:
                    self.coef[i] = (num + self.alpha/2) / den
                else:
                    self.coef[i] = 0                  
            if np.linalg.norm(self.coef - coef_old) < p*self.tol:
                break
            
        if self.fit_intercept:
            self.intercept = self.coef[0]
            self.coef = self.coef[1:]
        else:
            self.intercept = 0
        
        return self
        
    # Predicción.
    def predict(self, X):
        return self.intercept + X.dot(self.coef)
    
    # Coeficiente de determinación.
    def score(self, X, y):
        y_pred = self.predict(X)
        residual_sum_of_squares = np.sum((y - y_pred)**2)
        total_sum_of_squares = np.sum((y - np.mean(y))**2)
        return 1 - residual_sum_of_squares / total_sum_of_squares

# Grafica los coeficientes en función del parámetro de regularización.
# Puede utilizarse para Ridge y Lasso.
def coefficients_as_penalty_function(penalty, alphas, X, y, optimal = None, 
                                     **kwargs):
    coefs = []
    for alpha in alphas:
        estimator = penalty(alpha, **kwargs)
        coefs.append(estimator.fit(X, y).coef)

    plt.figure(figsize = (8, 7))
    plt.title('Coeficientes en función del parámetro de regularización')
    plt.xlabel('Parámetro de regularización')
    plt.ylabel('Coeficientes')
    plt.xscale('log')
    plt.plot(alphas, coefs)
    plt.legend(X.columns)
    
    if optimal is not None:
        plt.axvline(optimal, color = 'k')
    plt.show()

# Error cuadrático medio.
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Validación cruzada. Devuelve el score obtenido en cada iteración.
def cross_validate(estimator, X, y, scoring, n_splits):     
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    scores = np.empty(n_splits)
    indices = np.arange(len(X))
    subset = np.array_split(indices, n_splits)        
    for i in range(n_splits):
        Train = [index for index in indices if index not in subset[i]]
        Test = subset[i]
        estimator.fit(X[Train], y[Train])
        scores[i] = scoring(y[Test], estimator.predict(X[Test]))

    return scores
    
# Regresión Ridge por validación cruzada.
# Los métodos predict y score son heredados directamente de Ridge.
class RidgeCV(Ridge):
    # Constructor.
    def __init__(self, alphas, fit_intercept = True, n_splits = 5):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.n_splits = n_splits
    
    # Estima los parámetros.
    def fit(self, X, y):
        self.alpha = None
        self.best_score = None
        
        for alpha in self.alphas:
            estimator = Ridge(alpha, self.fit_intercept)
            score = np.mean(cross_validate(
                estimator, X, y, mean_squared_error, self.n_splits
            ))
            if self.best_score is None or score < self.best_score:
                self.alpha = alpha
                self.best_score = score
        
        estimator = Ridge(self.alpha, self.fit_intercept).fit(X, y)
        self.intercept = estimator.intercept
        self.coef = estimator.coef
        
        return self
    
# Regresión LASSO por validación cruzada.
# Los métodos predict y score son heredados directamente de Lasso.
class LassoCV(Lasso):
    # Constructor.
    def __init__(self, alphas, fit_intercept = True, n_splits = 5, max_iter = 
                 10000, tol = 1e-4):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
    
    # Estima los parámetros.
    def fit(self, X, y):
        self.alpha = None
        self.best_score = None
        
        for alpha in self.alphas:
            estimator = Lasso(alpha, self.fit_intercept, self.max_iter, 
                              self.tol)
            score = np.mean(cross_validate(
                estimator, X, y, mean_squared_error, self.n_splits
            ))
            if self.best_score is None or score < self.best_score:
                self.alpha = alpha
                self.best_score = score
                
        estimator = Lasso(self.alpha, self.fit_intercept, self.max_iter, 
                          self.tol).fit(X, y)   
        self.intercept = estimator.intercept
        self.coef = estimator.coef
        self.n_iter = estimator.n_iter
        
        return self