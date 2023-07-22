import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the trained model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('loan_approval_scaler.pkl')

def get_user_inputs():
    st.write("Please provide the following information:")
    age = st.number_input("Age:", value=30, min_value=18, max_value=100)
    income = st.number_input("Income:")
    employment_status = st.selectbox("Employment Status:", ['Employed', 'Self-Employed', 'Unemployed'])
    loan_amount = st.number_input("Loan Amount:")
    loan_purpose = st.selectbox("Loan Purpose:", ['Home', 'Personal', 'Education'])
    
    # Encode user inputs similar to the training dataset
    user_inputs = np.array([[age, income, employment_status, loan_amount, loan_purpose, 0, 0, 0]])  # Add three additional zeros for missing features

    # Load the LabelEncoders and StandardScaler used in Part 1
    label_encoder_employment = LabelEncoder()
    label_encoder_loan_purpose = LabelEncoder()
    scaler = StandardScaler()
    
    # Fit the LabelEncoders on the categorical features from the training data
    label_encoder_employment.fit(['Employed', 'Self-Employed', 'Unemployed'])
    label_encoder_loan_purpose.fit(['Home', 'Personal', 'Education'])

    # Transform the user inputs using the fitted LabelEncoders
    user_inputs[:, 2] = label_encoder_employment.transform([employment_status])
    user_inputs[:, 4] = label_encoder_loan_purpose.transform([loan_purpose])

    return user_inputs


# Function to predict loan approval
def predict_loan_approval(user_inputs):
    user_inputs_scaled = scaler.transform(user_inputs)

    # Make predictions using the loaded model
    prediction = model.predict(user_inputs_scaled)

    # Decode prediction (0: Not Approved, 1: Approved)
    label_encoder = LabelEncoder()
    label_encoder.fit(['N', 'Y'])  # Fit the encoder on the target variable 'Loan_Status'
    prediction_decoded = label_encoder.inverse_transform(prediction)

    return prediction_decoded[0]


# Main function
def main():
    st.title("Loan Approval Prediction")
    
    st.sidebar.image("celebal label cover picture.png", width=200, use_column_width='always')
    st.sidebar.title("Routhu Siddhartha")
    st.sidebar.markdown("Welcome to the Loan Prediction Model..!")
    st.sidebar.markdown(":e-mail: routhusiddhartha@gmail.com")
    st.sidebar.markdown("\U0001F476 [GitHub](https://github.com/siddhuR)")
    st.sidebar.markdown("\U0001F517 [LinkedIn](https://www.linkedin.com/in/siddhur/)")
    st.sidebar.markdown(":round_pushpin: Vijayawada, Andhra Pradesh")

    # About Me section with styled heading
    st.markdown("<h2 style='color: #ff5555;'>About Model</h2>", unsafe_allow_html=True)
    st.write(
        "Welcome to the fascinating world of Loan Approval Prediction! In this captivating project,"
         "we delve into the realm of machine learning to predict the likelihood of loan approval based on applicants' information."
         "Join us as we explore powerful algorithms and techniques that revolutionize the banking and lending industry,"
         "shaping the future of financial decision-making."
    )

    # Get user inputs
    user_inputs = get_user_inputs()

    if st.button("Predict Loan Approval"):
        # Predict and display the result
        prediction = predict_loan_approval(user_inputs)
        st.write(f"Loan Approval Prediction: {prediction}")

    # Contact section with styled heading
    st.markdown("<h2 style='color: #55aaff;'>Contact Me</h2>", unsafe_allow_html=True)
    st.write(
        "I'm always excited to work on new projects and collaborate with others. "
        "If you have any questions or would like to discuss a potential project, "
        "feel free to reach out to me at routhusiddhartha@gmail.com"
    )

    # Footer
    st.markdown("---")
    st.write("This portal is created with ❤️ using Streamlit by Routhu Siddhartha.")


# Rest of the code...

if __name__ == "__main__":
    main()
