
import pandas as pd 
from sklearn.model_selection import train_test_split
import joblib

save_data='src/data/'
######################
def main():
    # Load the dataset
    input="src/data/311_data_lat_lon.zip"
    df = pd.read_csv(input)
    # filter data to 2023 andd 2024.zip
    df=df[df.year>=2023]
    # new features
    df['month']=pd.DatetimeIndex(df['OPENEDDATETIME']).month.astype(object)
    df['year']=pd.DatetimeIndex(df['OPENEDDATETIME']).year.astype(object)
    df['day_of_week']=pd.DatetimeIndex(df['OPENEDDATETIME']).dayofweek.astype(object)
    df=df[['Category', 'SourceID','Council_District',
                'lat','lon','day_of_week','month','year','Late','holidays',
        'rain', 'tavg_f']].copy()    

    #------------
    numeric_features = ['tavg_f'] 
    categorical_features = ['Category', 'Council_District',
                'day_of_week','month','SourceID','rain','holidays'] 
    df[numeric_features]=df[numeric_features].astype(float)
    df[categorical_features]=df[categorical_features].astype(object) 
    # Define features and target variable
    features = ['Category', 'day_of_week','month', 'Council_District',
            'SourceID', 'holidays','rain','tavg_f']
    target = 'Late'
    # Split the dataset into features and target variable
    X = df[features]
    y = df[target].replace({'YES': 1, 'NO': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
              
    joblib.dump(X_train,f'{save_data}/X_train')
    joblib.dump(X_test,f'{save_data}/X_test')
    joblib.dump(y_train,f'{save_data}/y_train')
    joblib.dump(y_test,f'{save_data}/y_test')
    
if  __name__ == "__main__":
    main()    
