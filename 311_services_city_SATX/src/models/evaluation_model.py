import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
#
X_test=joblib.load('src/data/X_test')
y_test=joblib.load('src/data/y_test')
model=joblib.load('artifacts/model.pkl')
#

chosen_threshold =0.14
# Load the model from the file
###
def main():
    
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    # 
    predicted_classes = (y_pred_proba_test >= chosen_threshold).astype(int)
    print('the confusion matrix:')
    print(confusion_matrix(y_test,predicted_classes))
    print('the classification report:')
    print(classification_report(y_test,predicted_classes))

if __name__ == "__main__":
    # Run the main function
    main()

