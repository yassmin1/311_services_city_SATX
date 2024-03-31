import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import  train_test_split
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
    #save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_file_name='model.pkl'
    TRAINED_MODEL_DIR='./artifacts'
    save_path = f'{TRAINED_MODEL_DIR}/{save_file_name}'

    #remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
#
input_train_data='src/data'
X_train=joblib.load(f'{input_train_data}\X_train')
y_train=joblib.load(f'{input_train_data}\y_train')    

#------------
numeric_features = ['tavg_f'] 
categorical_features = ['Category', 'Council_District',
            'day_of_week','month','SourceID','rain','holidays'] 


def main():
    # Define preprocessing steps for numerical and categorical columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])



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
    parm={'learning_rate': 0.1, 'max_depth': 7,
    'min_child_weight': 1, 'n_estimators': 300, 'subsample': 0.9,'nthread':4,}
    # Define the model
    # Define the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', XGBClassifier(random_state=42,**parm))])

    #
 

    # Fit the pipeline on the entire dataset
    pipeline.fit(X_train, y_train) 

    # save
    save_pipeline(pipeline_to_persist=pipeline)
if __name__ == "__main__":
    # Run the main function
    main()
