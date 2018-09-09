import numpy as np
import pandas as pd 
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class RegressionModel():

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit(self):
        X = self.X.values
        y = self.y.values
        Q = np.linalg.inv(np.matmul(np.transpose(X), X))
        XXinvX = np.matmul(Q, np.transpose(X))
        self.coeffs = np.matmul(XXinvX, y)
        self.fit_errors = y-np.matmul(X, self.coeffs)
        self.s2 = sum(self.fit_errors**2)
        self.coeff_error = Q * self.s2

    def predict(self, X_test):
        return np.matmul(X_test.values, self.coeffs)


def main():
    customers = pd.read_csv('Ecommerce Customers')
    print(np.__version__)

    ones = np.ones(len(customers['Time on App']))
    customers['Constant'] = ones
    X = customers[['Constant', 'Avg. Session Length', 'Time on App', 'Time on Website',
    'Length of Membership']]
    y = customers['Yearly Amount Spent']
    customers.info()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

    lm = RegressionModel(X_train, y_train)
    lm.fit()
    coeffs = pd.DataFrame(lm.coeffs, index = lm.X.columns.values, columns = ['Coeff value'])
    coeffs['std'] = np.diag(lm.coeff_error)**0.5
    coeffs['p_value'] = (coeffs['Coeff value']/coeffs['std']).apply(lambda x: stats.norm.cdf(x))
    print(coeffs)
    plt.subplot(2,2,1)
    predictions = lm.predict(X_test)
    plt.scatter(predictions, y_test,cmap = 'viridis')
    plt.xlabel('predictions')
    plt.subplot(2,2,2)
    sns.distplot(lm.fit_errors)
    plt.xlabel('prediction distribution')
    plt.title(f'Error JB Statistic = {stats.jarque_bera(lm.fit_errors)}')
    plt.show()

if __name__ == '__main__':
    main()





