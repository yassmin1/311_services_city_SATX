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
    
    with st.expander("1. Ckeck if your request will be delayed or not  😀"):
        text_message = st.text_input("Please enter your message")
        if st.button("Predict"):
            prediction = spam_clf.predict(text_message]))

            if(prediction[0] == 0):
                info = 'NOT Delay'

            else:
                info = 'Delay'
            st.success('Prediction: {}'.format(info))

if __name__ == "__main__":
    main()