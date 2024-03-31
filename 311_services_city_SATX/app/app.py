import streamlit as st 
import joblib
#import os
#os.chdir("C:\\Users\\amanr\\OneDrive\\Desktop\\streamlit-spam-detector-main\\SSD")

# Load the model
spam_clf = joblib.load(open('artifacts/model.pkl','rb'))

# Load vectorizer
vectorizer = joblib.load('src/data/X_test')
### MAIN FUNCTION ###
def main(title = " Classification App for 311 Request SA".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 25px; color: blue;'>{}</h1>".format(title), 
    unsafe_allow_html=True)
    #st.image("images\myimage.png",width=100)
    info = ''
    # the order of feature:
    features=['Category', 'day_of_week','month', 'Council_District',
             'SourceID', 'holidays','rain','tavg_f']
   with st.expander("1. Check if your request will be delayed or not "):
        text_message = st.text_input("Please enter your request:")

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

        # ... (repeat for remaining features)

        # Input for Council_District (adjust based on input type)
        district_options = [ 1,  2,  3,  4,  5,  6,  7,  8,  9]
        selected_district = st.selectbox("District", district_options)
        features_dict["Council_District"] = selected_district
        #
        source_options = ['311 Mobile App', 'Constituent Call', 'Code Proactive Calls']
        selected_source = st.selectbox("District", source_options)
        features_dict["SourceID"] = selected_source
        #'holidays','rain','tavg_f'
       #
        features_dict["holiday"] = st.number_input("IsHoliday", format="%d")  # Display with one decimal place

        features_dict["rain"] = st.number_input("IsRaining", format="%d")  # Display with one decimal place
        features_dict["tavg_f"] = st.number_input("Average Temperature (Fahrenheit)", format="%f")  # Display with one decimal place


        # ... (repeat for remaining features, adjust input type as needed)

        if st.button("Predict"):
            # Prepare features list based on user input
            features_list = [features_dict[f] for f in features]

            # Make prediction using your model
            prediction = spam_clf.predict([features_list])

            if prediction[0] == 0:
                info = 'NOT Delay'
            else:
                info = 'Delay'

            st.success('Prediction: {}'.format(info))
        
if __name__ == "__main__":
    main()