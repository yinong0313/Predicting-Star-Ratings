import json
import gzip

# unzip file
with gzip.open('yelp_train_academic_dataset_business.json.gz') as f:
    data = [json.loads(line) for line in f]

star_ratings = [row['stars'] for row in data]

# city_model
from sklearn import base
from collections import defaultdict

# CityEstimator is based on avg stars of each city
class CityEstimator(base.BaseEstimator, base.RegressorMixin):

    def __init__(self):
        self.avg_stars = dict()

    def fit(self, X, y):
        # Store the average rating per city in self.avg_stars
        star_sum = defaultdict(int)
        count = defaultdict(int)
        for row, stars in zip(X, y):
            star_sum[row['city']] += stars
            count[row['city']] += 1
        for city in star_sum:
            # calculate average star rating and store in avg_stars
            self.avg_stars[city] = star_sum[city]/count[city]
        return self

    def predict(self, X):
        try:
            return [self.avg_stars[row['city']] for row in X]
        except KeyError as e:
            return [sum(self.avg_stars.values())/len(self.avg_stars)]*len(X)

city_est = CityEstimator()
city_est.fit(data, star_ratings)

# lat_long_model
# build a ColumnSelectTransformer that convert a list of dictionaries to an array containing selected keys of our feature matrix.
# based on KNN
import numpy as np
class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, X):
        column_data = []
        for row in X:
            column_data.append([row[col_names_fillin] for col_names_fillin in self.col_names])
        return column_data


from sklearn.pipeline import Pipeline
from sklearn import neighbors, model_selection

cv = model_selection.ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
scaled_nearest_neighbors = Pipeline([('cst', ColumnSelectTransformer(['latitude', 'longitude'])),
                                     ('neighbors', KNeighborsRegressor())])

param_grid = {"neighbors__n_neighbors": range(4, 50, 4)}    # parameters to Pipeline take the form [label]__[estimator_param]
scaled_nearest_neighbors_cv = model_selection.GridSearchCV(scaled_nearest_neighbors,
                                                       param_grid=param_grid, cv=10,
                                                       return_train_score=True)

scaled_nearest_neighbors_cv.fit(data, star_ratings)

# category_model
# deal with categorical features is one-hot encoding

from sklearn.feature_extraction import DictVectorizer
from collections import Counter

class DictEncoder(base.BaseEstimator, base.TransformerMixin):
    from sklearn.feature_extraction import DictVectorizer
    from collections import Counter

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X will come in as a list of lists of lists.  Return a list of
        # dictionaries corresponding to those inner lists.
        dict_list = []
        v = DictVectorizer(sparse=False)
        for nest_list in X:
            count_dict = Counter()
            for item in nest_list:
                for string in item:
                    count_dict[string] += 1
            dict_list.append(count_dict)
        X = v.fit_transform(dict_list)
        return v.inverse_transform(X)

# model based on ridge regression
from sklearn import pipeline
from sklearn import neighbors, model_selection
from sklearn.linear_model import Ridge

category_est = pipeline.Pipeline([('cst', ColumnSelectTransformer(['categories'])),('DE', DictEncoder()),
                                  ('DV',DictVectorizer()),("clf",Ridge())])

param_grid = {"clf__alpha": range(1, 10, 1)}    # parameters to Pipeline take the form [label]__[estimator_param]
category_est_cv = model_selection.GridSearchCV(category_est,
                                                       param_grid=param_grid, cv=5,
                                                       return_train_score=True)

category_est_cv.fit(data, star_ratings)

# attribute_model
#flattened dictionary, then one-hot encoding
class DictFlatten(base.BaseEstimator, base.TransformerMixin):
    from sklearn.feature_extraction import DictVectorizer
    from collections import Counter

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from collections import Counter
        att_list = []
        v = DictVectorizer(sparse=False)
        for nest_list in X:
            output_dict = Counter()
            for dict_att in nest_list:
                for key in dict_att:
                    if dict_att[key] == True and type(dict_att[key]) == bool:
                        output_dict[key] += 1
                    elif type(dict_att[key]) == str:
                        output_dict[key+'_'+dict_att[key]] += 1
                    elif type(dict_att[key]) == int:
                        output_dict[key+'_'+str(dict_att[key])] += 1
                    elif type(dict_att[key]) == dict:
                        first_step_dict = dict_att[key]
                        for key2 in first_step_dict:
                            if first_step_dict[key2] == True and type(first_step_dict[key2]) == bool:
                                output_dict[key+'_'+key2] += 1
                            elif type(first_step_dict[key2]) == str:
                                output_dict[key+'_'+key2+first_step_dict[key2]] += 1
                            elif type(first_step_dict[key2]) == int:
                                output_dict[key+'_'+key2+str(first_step_dict[key2])] += 1
            att_list.append(output_dict)
        X_dict= v.fit_transform(att_list)
        return v.inverse_transform(X_dict)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# linear fitting
attributes_est = pipeline.Pipeline([('cst', ColumnSelectTransformer(['attributes'])),('DF', DictFlatten()),
                                   ('DV',DictVectorizer()),('clf', Ridge())])

param_grid = {"clf__alpha": range(6, 10, 1)}
attributes_est_cv = model_selection.GridSearchCV(attributes_est,
                                                       cv=5,
                                                        param_grid = param_grid,
                                                         n_jobs=2,
                                                       return_train_score=True)

#attributes_est_cv.fit(data, star_ratings)

# non-linear fitting (fit residuals)
predictions = attributes_est_cv.predict(data)
residuals= star_ratings-predictions

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

attributes_est2 = pipeline.Pipeline([('cst', ColumnSelectTransformer(['attributes'])),('DF', DictFlatten()),
                                   ('DV',DictVectorizer()),('svr', SVR())])

param_grid = {"svr__kernel": ['poly'],"svr__degree":range(1, 6, 1),"svr__coef0":range(1, 3, 1),"svr__epsilon":[0.3]}
attributes_est2_cv = model_selection.GridSearchCV(attributes_est2,
                                                       cv=5,
                                                        param_grid = param_grid,
                                                         n_jobs=2,
                                                  return_train_score=True
                                                       )

#attributes_est2_cv.fit(data,residuals)

# linear + non-linear fitting
class AttributesEstimator(base.BaseEstimator, base.RegressorMixin):

    def __init__(self, linear_model, non_linear_model):
        self.linear_model = linear_model
        self.non_linear_model = non_linear_model

    def fit(self,X,y):
        self.linear_model.fit(X, y)
        residuals= y-self.linear_model.predict(X)
        self.non_linear_model.fit(X,residuals)
        return self

    def predict(self, X):
        return (self.linear_model.predict(X) + self.non_linear_model.predict(X))

attributes_est_tot = AttributesEstimator(attributes_est_cv,attributes_est2_cv)
attributes_est_tot.fit(data,star_ratings)

#full model (combine the above models together)
class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X,y)
        return self

    def transform(self, X):
        prediction = self.estimator.predict(X)
        return (np.asarray(prediction).reshape(-1,1))

from sklearn.pipeline import FeatureUnion

union = FeatureUnion([('city_model', EstimatorTransformer(city_est)),
  ('lat_long_model', EstimatorTransformer(scaled_nearest_neighbors_cv)),
  ('category_model', EstimatorTransformer(category_est_cv)),
  ('attribute_model', EstimatorTransformer(attributes_est_tot))

        # FeatureUnions use the same syntax as Pipelines
    ])

# final model
k_union = pipeline.Pipeline([
  ("features", union),
  ("linreg", LinearRegression(fit_intercept=True))
  ])
k_union.fit(data, star_ratings)
