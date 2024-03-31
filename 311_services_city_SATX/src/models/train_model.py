import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from xgboost import XGBClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, classification_report,roc_curve,auc
from sklearn.base import BaseEstimator
import joblib

#------
def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_file_name='model.pkl'
    TRAINED_MODEL_DIR='../../models'
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)

#---------------


class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator=RandomForestClassifier()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier (default is RandomForestClassifier)
        """

        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)

# Load the dataset
input="../data/311_data_lat_lon.csv"
df = pd.read_csv(input)
# filter data to 2023 andd 2024
df=df[df.year>=2023]
# new features
df['month']=pd.DatetimeIndex(df['OPENEDDATETIME']).month.astype(object)
df['year']=pd.DatetimeIndex(df['OPENEDDATETIME']).year.astype(object)
df['day_of_week']=pd.DatetimeIndex(df['OPENEDDATETIME']).dayofweek.astype(object)


df=df[['Category', 'SourceID','Council_District',
             'lat','lon','day_of_week','month','year','Late','holidays',
       'rain', 'tavg_f']].copy()

 #------------
df[numeric_features]=df[numeric_features].astype(float)
df[categorical_features]=df[categorical_features].astype(object) 
# Define features and target variable
features = ['Category', 'day_of_week','month', 'Council_District',
            'SourceID', 'holidays','rain','tavg_f']
target = 'Late'
# Split the dataset into features and target variable
X = df[features]
y = df[target].replace({'YES': 1, 'NO': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)           
#----------------

def main():
    # Define preprocessing steps for numerical and categorical columns
    numeric_features = ['tavg_f']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_features = ['Category', 'Council_District',
            'day_of_week','month','SourceID','rain','holidays']

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', XGBClassifier(random_state=42))])

    #
 

    parm={'learning_rate': 0.1, 'max_depth': 7,
    'min_child_weight': 1, 'n_estimators': 300, 'subsample': 0.9,'nthread':4,}
    # Define the model
    model=XGBClassifier(**parm )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', model)])
 
    # Fit the pipeline on the entire dataset
    pipeline.fit(X_train, y_train) 

    # save
    save_pipeline(pipeline_to_persist=pipeline)
if __name__ == "__main__":
    # Run the main function
    main()
