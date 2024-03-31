import joblib
import pandas as pd
import json
from sklearn.metrics import confusion_matrix,classification_report
#
#  features = ['Category', 'day_of_week','month', 'Council_District',
#            'SourceID', 'holidays','rain','tavg_f']
            
X_test=joblib.load('src/data/X_test')[:1]
model=joblib.load('artifacts/model.pkl')
#####
chosen_threshold =0.14
###
def main(): 
    print(f'{X_test}')   
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    predicted_classes = (y_pred_proba_test >= chosen_threshold).astype(int)
    print(f'the probability value: {y_pred_proba_test}')
    print(f'the predicted class result:{predicted_classes}')

if __name__ == "__main__":
    # Run the main function
    main()

