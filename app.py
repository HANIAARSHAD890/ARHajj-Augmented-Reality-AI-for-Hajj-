import streamlit as st
import pandas as pd
import joblib
import gemini_integration as gemini

# Load trained model
model = joblib.load("decision_tree_model.pkl")

st.title("🕋 Haji Health Predictor")

# Define input options
activity_types = ["Prayer", "Resting", "Sa’i", "Tawaf", "Walking"]
weather_conditions = ["Clear", "Cloudy", "Rainy"]
ar_interaction = ["Completed", "In Progress", "Started"]
emergency_event = ["Yes", "No"]
age_groups = ["18-30", "31-50", "51-70", "70+"]
nationalities = ["Egyptian", "Indian", "Indonesian", "Other", "Pakistani", "Saudi"]
transport_modes = ["Bus", "Car", "Train", "Walking"]
incident_types = ["Lost Item", "Medical Emergency", "Security Breach", "Theft", "Unruly Behavior"]
event_types = ["Crowd Congestion", "Medical Emergency", "Religious Activity", "Transport Delay"]
pilgrim_experience = ["Experienced", "First-Time"]
crowd_morale = ["Negative", "Neutral", "Positive"]
ar_nav_success = ["Yes", "No"]

# Categorical inputs
activity = st.selectbox("Activity Type", activity_types)
weather = st.selectbox("Weather Condition", weather_conditions)
ar_state = st.selectbox("AR System Interaction", ar_interaction)
emergency = st.selectbox("Emergency Event", emergency_event)
age_group = st.selectbox("Age Group", age_groups)
nationality = st.selectbox("Nationality", nationalities)
transport = st.selectbox("Transport Mode", transport_modes)
incident = st.selectbox("Incident Type", incident_types)
event_type = st.selectbox("Event Type", event_types)
experience = st.selectbox("Pilgrim Experience", pilgrim_experience)
morale = st.selectbox("Crowd Morale", crowd_morale)
nav_success = st.selectbox("AR Navigation Success", ar_nav_success)

# Numerical inputs
fatigue = st.slider("Fatigue Level", 0, 10, 5)
stress = st.slider("Stress Level", 0, 10, 5)
crowd_density = st.number_input("Crowd Density", min_value=0.0, value=1.0)
lat = st.number_input("Location Latitude", value=21.4)
lon = st.number_input("Location Longitude", value=39.8)
speed = st.number_input("Movement Speed", min_value=0.0, value=1.0)
temperature = st.slider("Temperature (°C)", 25.0, 50.0, 37.0)
sound = st.slider("Sound Level (dB)", 30.0, 120.0, 70.0)
queue_time = st.number_input("Queue Time (min)", min_value=0, value=5)
waiting_transport = st.number_input("Waiting Time for Transport (min)", min_value=0, value=10)
security_wait = st.number_input("Security Checkpoint Wait (min)", min_value=0, value=5)
interaction_freq = st.number_input("Interaction Frequency", min_value=0, value=2)
distance = st.number_input("Distance Between People (m)", min_value=0.0, value=1.0)
time_spent = st.number_input("Time Spent at Location (min)", min_value=0, value=20)
safety_rating = st.slider("Perceived Safety Rating", 1, 5, 3)

# Prepare full feature list
all_features = [  # Your full one-hot feature list
    'Activity_Type_Prayer', 'Activity_Type_Resting', 'Activity_Type_Sa’i', 'Activity_Type_Tawaf', 'Activity_Type_Walking',
    'Weather_Conditions_Clear', 'Weather_Conditions_Cloudy', 'Weather_Conditions_Rainy',
    'AR_System_Interaction_Completed', 'AR_System_Interaction_In Progress', 'AR_System_Interaction_Started',
    'Emergency_Event_No', 'Emergency_Event_Yes',
    'Age_Group_18-30', 'Age_Group_31-50', 'Age_Group_51-70', 'Age_Group_70+',
    'Nationality_Egyptian', 'Nationality_Indian', 'Nationality_Indonesian', 'Nationality_Other', 'Nationality_Pakistani', 'Nationality_Saudi',
    'Transport_Mode_Bus', 'Transport_Mode_Car', 'Transport_Mode_Train', 'Transport_Mode_Walking',
    'Incident_Type_Lost Item', 'Incident_Type_Medical Emergency', 'Incident_Type_Security Breach', 'Incident_Type_Theft', 'Incident_Type_Unruly Behavior',
    'Event_Type_Crowd Congestion', 'Event_Type_Medical Emergency', 'Event_Type_Religious Activity', 'Event_Type_Transport Delay',
    'Pilgrim_Experience_Experienced', 'Pilgrim_Experience_First-Time',
    'Crowd_Morale_Negative', 'Crowd_Morale_Neutral', 'Crowd_Morale_Positive',
    'AR_Navigation_Success_No', 'AR_Navigation_Success_Yes',
    'Fatigue_Level', 'Stress_Level', 'Crowd_Density', 'Location_Lat', 'Location_Long', 'Movement_Speed', 'Temperature',
    'Sound_Level_dB', 'Queue_Time_minutes', 'Waiting_Time_for_Transport', 'Security_Checkpoint_Wait_Time',
    'Interaction_Frequency', 'Distance_Between_People_m', 'Time_Spent_at_Location_minutes', 'Perceived_Safety_Rating'
]

# Initialize input vector
input_data = dict.fromkeys(all_features, 0)

# Set one-hot categorical values
input_data[f"Activity_Type_{activity}"] = 1
input_data[f"Weather_Conditions_{weather}"] = 1
input_data[f"AR_System_Interaction_{ar_state}"] = 1
input_data[f"Emergency_Event_{emergency}"] = 1
input_data[f"Age_Group_{age_group}"] = 1
input_data[f"Nationality_{nationality}"] = 1
input_data[f"Transport_Mode_{transport}"] = 1
input_data[f"Incident_Type_{incident}"] = 1
input_data[f"Event_Type_{event_type}"] = 1
input_data[f"Pilgrim_Experience_{experience}"] = 1
input_data[f"Crowd_Morale_{morale}"] = 1
input_data[f"AR_Navigation_Success_{nav_success}"] = 1

# Set numerical values
input_data["Fatigue_Level"] = fatigue
input_data["Stress_Level"] = stress
input_data["Crowd_Density"] = crowd_density
input_data["Location_Lat"] = lat
input_data["Location_Long"] = lon
input_data["Movement_Speed"] = speed
input_data["Temperature"] = temperature
input_data["Sound_Level_dB"] = sound
input_data["Queue_Time_minutes"] = queue_time
input_data["Waiting_Time_for_Transport"] = waiting_transport
input_data["Security_Checkpoint_Wait_Time"] = security_wait
input_data["Interaction_Frequency"] = interaction_freq
input_data["Distance_Between_People_m"] = distance
input_data["Time_Spent_at_Location_minutes"] = time_spent
input_data["Perceived_Safety_Rating"] = safety_rating

# When user clicks Predict
if st.button("Predict Health Condition"):
    input_df = pd.DataFrame([input_data])
    
    prediction = model.predict(input_df)[0]
    if prediction in ["Injured", "Heatstroke"]:
        st.error(f"🚨 Critical Condition Detected: **{prediction}**")
    else:
        st.success(f"🏥 Predicted Health Condition: **{prediction}**")
    st.subheader("🤖 Gemini Health Advisory")
    prompt = f"A pilgrim has been predicted to be in the condition: {prediction}. What advice, precautions, or support should be given?"
    gemini_response = gemini.query_gemini(prompt)
    st.write(gemini_response)