import streamlit as st
import pickle
import os
import pandas as pd


def load_model():
    with open("personality_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model
model = load_model()
logo_path = "pulogo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=350)


st.title("Personality Prediction Web App (Are you an introvert or extrovert?)")
st.write("Logistic Regression Model with Pipeline")
st.markdown("Please enter the behavioral details and time spent below to predict the personality.")


logo_path = "mpt2.jpg"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)

st.sidebar.header("Midterm Project")
st.sidebar.markdown("**Student Name:** May Phuu Thwel")
st.sidebar.markdown("**Major:**Statistics and Data Science")
st.sidebar.markdown("**Course Name:** Introduction to Machine Learning")
st.sidebar.markdown("**Instructor Name:** Professor Nwe Nwe Htay Win")

Time_spent_Alone = st.number_input("Time spent alone per day (in hours)", min_value = 0.0, max_value = 11.0)
Social_event_attendance = st.slider("Social event attendance score per month", min_value = 0.0, max_value = 10.0)
Going_outside = st.slider("How often do you go outside per week?",min_value = 0.0, max_value = 7.0)
Friends_circle_size = st.number_input("How many close friends do you have in general?",min_value = 0.0, max_value = 15.0)
Post_frequency = st.number_input("How many posts do you upload on social media per week?",min_value = 0.0, max_value = 10.0)


Stage_fear = st.radio("Do you have Stage Fear?", ["Yes","No"])
Drained_after_socializing = st.selectbox("Do you feel drained after socializing?",["Yes","No"])

user_data ={
    "Time_spent_Alone": Time_spent_Alone,
    "Stage_fear": Stage_fear,
    "Social_event_attendance": Social_event_attendance,
    "Going_outside": Going_outside,
    "Drained_after_socializing": Drained_after_socializing,
    "Friends_circle_size": Friends_circle_size,
    "Post_frequency": Post_frequency
}

p_image = {
    'Introvert': 'Introvert.jpg',
    'Extrovert': 'Extrovert.jpg'
}

input_df = pd.DataFrame([user_data])
recommendations = {
    "Introvert": "As with more percent on introversion, to recharge your energy, consider setting aside quiet time for yourself, engage in meaningful one-on-one conversations, and don't feel pressured to attend every social event. Focus on deepening your close relationships, do your solo favorite activites. Find some interesting activites with this article: https://introvertdear.com/news/14-introvert-friendly-activities-to-do-alone-or-with-a-small-group/",
    "Extrovert": "As with more percent on extroversion, to boost your energy, make sure to balance social activities with some alone time to recharge. Take opportunities to lead group activities and expand your social network. Try some hobbies from this article: https://www.underhobby.net/the-best-hobbies-for-extroverts/ "
}
if st.button("Predict Personality"):
    try:
        model = load_model()
        prediction = model.predict(input_df)[0]
        st.subheader("Predicted Personality:")
        st.success(f"**{prediction}**")
        st.image(p_image[prediction], caption = f"{prediction}, width = 200")
        st.subheader("Recommendation:")
        st.info(recommendations[prediction])
        
    except Exception as e:
        st.error(f" Prediction failed: {e}")

        st.write("Please check that the model and inputs match the training columns.")
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]
st.write("### Prediction Probability")
st.progress(probability)
st.write(f"{probability*100:.2f}% chance of being {prediction}")


























