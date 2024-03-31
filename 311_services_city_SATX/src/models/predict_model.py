import joblib
import pandas as pd
#
X_test=pd.read_csv('../data/X_test')

#

chosen_threshold =0.14
# Load the model from the file
###
def main():
    model=joblib.load(model, 'model.pkl')
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    # 
    predicted_classes = (y_pred_proba_test >= chosen_threshold).astype(int)
    print(confusion_matrix(y_test,predicted_classes))
    print(classification_report(y_test,predicted_classes))

if __name__ == "__main__":
    # Run the main function
    main()

