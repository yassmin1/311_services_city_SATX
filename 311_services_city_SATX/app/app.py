import streamlit as st 
import joblib
import pandas as pd 
#import os
#os.chdir("C:\\Users\\amanr\\OneDrive\\Desktop\\streamlit-spam-detector-main\\SSD")

# Load the model
delay_clf = joblib.load(open('artifacts/model.pkl','rb'))

# Load vectorizer
vectorizer = joblib.load('src/data/X_test')
### MAIN FUNCTION ###
def main(title = " App for 311 Service Request SA.TX".upper()):
     # Ensure proper cache handling
    st.cache(suppress_st_warning=True)
    st.set_page_config( page_icon=":bar_chart:", layout="centered")
    st.markdown("<h1 style='text-align: center; font-size: 45px; color: black;'>{}</h1>".format(title), 
    unsafe_allow_html=True)
    # Add custom CSS to style labels and other elements   
    custom_css = """
    .stSelectbox label {
        font-size: 35px;
        color:green;/* Adjust as desired */
    }
       body {
        margin: 0;
        padding: 10px;
    }
    .stSelectbox, .stButton {  /* Target specific elements */
        font-size:30px;
        margin: 5px;
        padding: 5px;
        
    }
    
    
    """
    st.write(f'<style>{custom_css}</style>', unsafe_allow_html=True) 
  
    ##
    #st.image("images\myimage.png",width=100)
    info = ''
    # the order of feature:
    features=['Category', 'day_of_week','month', 'Council_District',
             'SourceID', 'holidays','rain','tavg_f']

    # Create dictionary to store all features
    features_dict = {}

    # Category input
    category_options = ['Dockless Vehicle', 'Streets & Infrastructure', 'Animals',
    'Property Maintenance', 'Traffic Signals and Signs', 'Parks',
    'Health & Sanitation', 'Information', 'Historic Preservation',
    'Solid Waste Services', 'Graffiti']  
    selected_category = st.selectbox("Category", category_options)
    features_dict["Category"] = selected_category

    # Day of week input
    day_of_week_options = [0, 1, 2, 3, 4, 5,6]# 0 is monday
    selected_day = st.selectbox("Day of week", day_of_week_options)
    features_dict["day_of_week"] = selected_day

    # Month input (adjust based on your data format)
    month_options = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
    selected_month = st.selectbox("Month", month_options)
    features_dict["month"] = selected_month

    # Input for Council_District (adjust based on input type)
    district_options = [ 1,  2,  3,  4,  5,  6,  7,  8,  9]
    selected_district = st.selectbox("District", district_options)
    features_dict["Council_District"] = selected_district
    #
    source_options = ['311 Mobile App', 'Constituent Call', 'Code Proactive Calls']
    selected_source = st.selectbox("Source", source_options)
    features_dict["SourceID"] = selected_source
    #'holidays','rain','tavg_f'
    #
    features_dict["tavg_f"] = st.number_input("Average Temperature (Fahrenheit)", format="%f")  # Display with one decimal place
    # Features with 0/1 values using limited selectbox options
    zero_one_options = ["0", "1"]  # "No" represents 0, "Yes" represents 1
    features_dict["holidays"] = st.selectbox("Holidays", zero_one_options)
    features_dict["rain"] = st.selectbox("Rain", zero_one_options)
    

    # ... (repeat for remaining features, adjust input type as needed)

    if st.button("Predict"):
        # Prepare features list based on user input
        features_list = [features_dict[f] for f in features]
        print(features_list)
        df_features=pd.DataFrame(dict(zip(features,features_list)),index=[0]).fillna(0)
        print(df_features)
        # Make prediction using your model
        prediction = delay_clf.predict_proba(df_features[:1])[:,1]
        print(f'prediction_proba={prediction}')
        # Prediction output with default styling
        info = 'Delay' if prediction[0] >= 0.14 else 'NO Delay'
        st.success(info)
        
if __name__ == "__main__":
    main()