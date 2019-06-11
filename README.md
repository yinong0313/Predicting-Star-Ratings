# Predicting-Star-Ratings

The goal of this project is to predict a new venue's popularity from information available when the venue opens. This project is preformed by machine learning from a data set of venue popularities provided by Yelp. 

The data set contains meta data about the venue (where it is located, the type of food served, etc.). It also contains a star rating.

More detail of how the code work can be found in `project_details.pdf`

# The prediction consists four models

## Model 1: city_model.
- A custom estimator that based solely on the city of a venue (average star of a city)

## Model 2: lat_long_model.
- Use the latitude and longitude of a venue as features to understand neighborhood dynamics
- A ColumnSelectTransformer is used to transform latitude and longitude values to an array containing selected keys of feature matrix

## Model 3: category_model.
- An estimator that considers the categories is built. 
- Applied one-hot encoding (DictVectorizer) to deal with categorical features.

## Model 4: attribute_model.
- An estimator based on these attributes (e.g., Attire, Ambiance, Good for, Noise level, Caters…)
- Firstly, flatten the nested dictionary to a single level, then encode them with one-hot encoding

- Run `build_model.py`

It will output the final prediction model.
