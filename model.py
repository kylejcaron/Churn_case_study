import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import confusion_matrix

def plot_loss(clf, params, X_test, y_test):
    '''Plot training deviance.  Stolen from sklearn documentation'''    
    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)

    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel(clf.loss)

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

class Cleaning():
	def __init__(self):
		pass

	def fit(self, df):
		self.df = df

	def dt_and_churn(self):
		copy = self.df.copy()
		copy['last_trip_datetime'] = pd.to_datetime(copy["last_trip_date"])
		copy["signup_date_datetime"] = pd.to_datetime(copy["signup_date"])
		copy = copy.drop(columns=["last_trip_date", "signup_date"])
		copy["churn"] = copy['last_trip_datetime'].dt.month < 6
		copy.churn = copy.churn.astype(int)
		return copy

	# def driver_rating(self): ##add median
	# 	copy = self.df.copy()
	# 	val = copy['avg_rating_of_driver'][copy['avg_rating_of_driver'] != np.NaN].median()
	# 	copy['avg_rating_of_driver'] = copy['avg_rating_of_driver'].replace(np.NaN, val)
	# 	return copy
	
	def driver_rating(self):
		copy = self.df.copy()
		copy['rating_5'] = (copy.avg_rating_of_driver == 5)*1
		copy['no_rating_of_driver'] = pd.isnull(copy.avg_rating_of_driver)*1
		# Bin ratings
		copy['bin_avg_rating_by_driver'] = pd.cut(copy.avg_rating_by_driver, bins=[0., 2.99, 3.99, 4.99, 5], 
                                        include_lowest=True, right=True)
		copy['bin_avg_rating_of_driver'] = pd.cut(copy.avg_rating_of_driver, bins=[0., 2.99, 3.99, 4.99, 5], 
                                        include_lowest=True, right=True)

		copy.bin_avg_rating_by_driver.cat.add_categories('missing', inplace=True)
		copy.bin_avg_rating_of_driver.cat.add_categories('missing', inplace=True)

		copy['bin_avg_rating_by_driver'].fillna('missing', inplace=True)
		copy['bin_avg_rating_of_driver'].fillna('missing', inplace=True)
		copy.drop(['avg_rating_of_driver', 'avg_rating_by_driver'], inplace=True, axis=1)
		return copy


	def dummy(self):
		copy = self.df.copy()
		copy.luxury_car_user = copy.luxury_car_user.astype(int)
		new = pd.get_dummies(copy[['city', 'phone', 'luxury_car_user', 
			'bin_avg_rating_of_driver', 'bin_avg_rating_by_driver']], drop_first=True)
		copy = copy.drop(['city', 'phone', 'luxury_car_user', 
			'bin_avg_rating_of_driver', 'bin_avg_rating_by_driver'], axis=1)
		newdf = pd.concat([copy, new], axis=1)
		return newdf

	def scaler(self):
		copy = self.df.copy()
		X = copy[['avg_dist', 'avg_surge', 'surge_pct', 'trips_in_first_30_days',
    		'weekday_pct']].values
		scale = StandardScaler()
		scale.fit(X)
		copy[['avg_dist', 'avg_surge', 'surge_pct', 'trips_in_first_30_days',
    		'weekday_pct']] = scale.transform(X)
		return copy
	
	def drop_dt(self):
		copy = self.df.copy()
		copy = copy.drop(['last_trip_datetime', 'signup_date_datetime'], axis=1)
		return copy

	def all(self, scale=False):
		self.df = self.dt_and_churn()
		self.df = self.driver_rating()
		self.df.phone = self.df.phone.fillna('no_phone')
		self.df = self.df.dropna(axis=0)
		self.df = self.dummy()
		if scale==True:
			self.df = self.scaler()
		return self.df

class EDA():
	def __init__(self):
		pass

	def fit(self, df):
		self.df = df
		return self

	def pie_plot(self, col):
		counts = self.df[col].value_counts()
		plt.pie(counts, labels=counts.index)
		plt.title(col)
		plt.show()

	def cat_charter(self):
		copy = self.df.copy()
		newdf = np.empty((len(self.df.columns),2))
		copy = copy.replace('None or Unspecified', np.NaN)
		for i, column in enumerate(copy.columns):
			appends =[len(copy[column].unique()), copy[column].count()] 
			newdf[i,:] = appends
		result = pd.DataFrame(newdf, index =[name for name in copy.columns], 
			columns = ['Unique_Values', 'Non_null_count'])
		return result

	def scatplot(self, col1, col2):
		plt.scatter(self.df[col1], self.df[col2])
		plt.xlabel(col1)
		plt.ylabel(col2)
		plt.show()

class Model():
	def __init__(self):
		pass
	
	def fit(self, df):
		self.df = df
		self.y = self.df.churn.values
		self.X = self.df.drop('churn', axis=1).values
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
			self.y, test_size=0.2)
		self.names = df.drop('churn',axis=1).columns
		return self
	
	def CV(self, algorithm):
		cv = cross_val_score(algorithm, self.X_train, self.y_train, n_jobs = -1, 
			scoring='roc_auc', cv=10)
		return cv

	def fit_algorithm(self, algorithm):
		alg = algorithm()
		alg.fit(X_train, y_train)
		alg.score()
	
	def importance(self, algorithm):
		feature_importances = algorithm.feature_importances_
		top10_colindex = np.argsort(feature_importances)[::-1][0:10]
		#names = self.names[top10_colindex]
		# print(names)
		feature_importances = feature_importances[top10_colindex]
		feature_importances = feature_importances / np.sum(feature_importances)
		y_ind = np.arange(9, -1, -1) # 9 to 0
		fig = plt.figure(figsize=(8, 8))
		plt.barh(y_ind, feature_importances, height = 0.3, align='center')
		plt.ylim(y_ind.min() + 0.5, y_ind.max() + 0.5)
		plt.yticks(y_ind, [self.names[i] for i in top10_colindex])
		plt.xlabel('Relative feature importances')
		plt.ylabel('Features')
	
	def plot_partial_dependencies(self, algorithm):
		feature_importances = algorithm.feature_importances_
		top10_colindex = np.argsort(feature_importances)[::-1][0:10]
		plot_partial_dependence(algorithm, self.X_train, features=top10_colindex, feature_names = self.names, figsize=(12,10))
		plt.tight_layout()












