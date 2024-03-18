# https://www.michigandaily.com/news/academics/we-looked-301-high-schools-most-applicants-u-m-heres-what-we-found/
# https://www.michigandaily.com/news/academics/across-country-or-across-state-undergraduates-come-u-varying-experiences/
# https://www.mlive.com/news/erry-2018/05/3f2ff604553639/see_where_michigan_colleges_ge.html#:~:text=University%20of%20Michigan%3A%202%2C875&text=Class%20of%202017.-,Here's%20a%20list%20of%20the%20public%20schools%20sending%20the%20most,High%20School%3A%2078%2C%2016%25%3B
# https://obp.umich.edu/wp-content/uploads/pubdata/cds/cds_2022-2023_umaa.pdf

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# Load your CSV file into a pandas DataFrame
df = pd.read_csv('feeder_schools.csv')

# Select only the columns of interest
selected_columns = ['School', 'State', 'Acceptance Rate', 'Mean SAT', 'Median Income']
df = df[selected_columns]

# Separate features (X) and target variable (y)
X = df.drop('Acceptance Rate', axis=1)
y = df['Acceptance Rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the columns based on data types
categorical_cols = ['School', 'State']
quantitative_cols = ['Mean SAT', 'Median Income']

# Create transformers for preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

quantitative_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=20)),  # You can adjust the number of neighbors as needed
    ('scaler', StandardScaler())
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', quantitative_transformer, quantitative_cols)
    ])

# Create a pipeline with the preprocessor and the linear regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Save the model as a pickle file
with open('linear_regression_model_knn.pkl', 'wb') as file:
    pickle.dump(model, file)
