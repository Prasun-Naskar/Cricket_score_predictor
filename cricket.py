import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


# loading the saved models

ipl_model = pickle.load(
    open('ipl.pkl', 'rb'))

odi_model = pickle.load(
    open('odi.pkl', 'rb'))

t20_model = pickle.load(
    open('t20.pkl', 'rb'))

teams_odi = [
    'Australia',
    'Pakistan',
    'Afghanistan',
    'Zimbabwe',
    'New Zealand',
    'Bangladesh',
    'South Africa',
    'India',
    'England',
    'Sri Lanka',
    'West Indies',
    'Ireland',
    'Namibia',
    'Netherlands'
    ]

cities_odi=['Brisbane','Perth', 'Canberra', 'Edinburgh', 'Christchurch', 'Nelson'
 'Auckland', 'Hamilton', 'Wellington', 'London', 'Birmingham','Cardiff',
 'Hong Kong', 'Antigua', 'Barbados', 'Bristol', 'Mirpur', 'Chittagong',
 'Dharmasala', 'Delhi', 'Chandigarh', 'Ranchi', 'Visakhapatnam', 'Leeds',
 'Southampton', 'Manchester', 'Nottingham', 'Dublin', 'Pune', 'Cuttack',
 'Kolkata','Greater Noida', 'Abu Dhabi', 'Bulawayo', 'Kimberley', 'Paarl',
 'East London', 'Guyana', 'Dubai', 'Colombo', 'St Lucia', 'Trinidad', 'Jamaica',
 'Hambantota', 'Mount Maunganui', 'Dunedin', 'Whangarei', 'Indore', 'Bengaluru',
 'Nagpur', 'Chester-le-Street', 'Mumbai' ,'Kanpur', 'Durban', 'Centurion',
 'Cape Town', 'Johannesburg', 'Port Elizabeth', 'Dharamsala', 'Dhaka', 'Harare',
 'Bloemfontein', 'Taunton', 'Hobart', 'Sydney', 'Adelaide', 'Melbourne',
 'St Kitts', 'Belfast', 'Sylhet', 'Chattogram', 'Napier', 'Kuala Lumpur',
 'Guwahati', 'Thiruvananthapuram', 'Bridgetown', "St George's", 'Gros Islet',
 'Dehra Dun', 'Hyderabad', 'Bready', 'Sharjah', 'Windhoek', 'Potchefstroom',
 'Chennai', 'Rajkot', 'Deventer', 'Providence', 'Port of Spain', 'Lucknow',
 'Aberdeen', 'Karachi', 'Lauderhill', 'Al Amarat', 'Kandy', 'Kirtipur',
 'St Vincent', 'Canterbury', 'Ahmedabad', 'Vadodara', 'Lahore', 'Queenstown',
 'Peshawar', 'Bogra', 'Fatullah', 'Faridabad', 'Margao', 'Jamshedpur', 'Grenada',
 'Nairobi', 'Jaipur', 'Faisalabad', 'Benoni', 'Glasgow', 'Bangalore', 'Kochi',
 'Gwalior', 'King City', 'Ayr', 'Sind', 'Darwin', 'Dominica', 'Toronto',
 'Amstelveen', 'Skating and Curling Club', 'Khulna', 'Lincoln']

cities_ipl=['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bengaluru', 'Mumbai', 'Kolkata',
 'Bangalore', 'Delhi' 'Chandigarh' 'Kanpur' 'Chennai' 'Jaipur'
 'Visakhapatnam','Abu Dhabi', 'Cape Town', 'Port Elizabeth', 'Durban',
 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
 'Cuttack', 'Ahmedabad', 'Nagpur', 'Dharamsala', 'Kochi', 'Raipur', 'Ranchi']

cities_t20=['Victoria', 'Napier','Mount Maunganui', 'Auckland', 'Southampton',
 'Taunton', 'Cardiff', 'Chester-le-Street', 'Kanpur', 'Nagpur', 'Bangalore',
 'Lauderhill', 'Abu Dhabi', 'Hobart', 'Wellington', 'Hamilton', 'Bloemfontein',
 'Potchefstroom', 'Barbados', 'Trinidad', 'Colombo', 'St Kitts', 'Jamaica',
 'Nelson', 'Ranchi', 'Birmingham','Manchester', 'Bristol','Delhi', 'Rajkot',
 'Thiruvananthapuram', 'Lahore', 'Johannesburg', 'Centurion', 'Cape Town',
 'Cuttack', 'Indore','Mumbai', 'Dhaka', 'Karachi', 'Brisbane', 'Dehradun',
 'Sylhet', 'Kolkata', 'Lucknow', 'Chennai', 'Gros Islet', 'Basseterre',
 'Visakhapatnam', 'Bengaluru', 'Adelaide', 'Melbourne', 'Sydney', 'Canberra',
 'Perth', 'East London', 'Durban','Port Elizabeth', 'Chandigarh', 'Hyderabad',
 'Christchurch', 'Providence','Kandy', 'Chattogram', 'Pune', 'Paarl', 'London',
 'Nairobi', 'Nottingham', 'King City', 'Guyana', 'St Lucia', 'Antigua', 'Mirpur',
 'Hambantota', 'Ahmedabad', 'St Vincent', 'Chittagong', 'Dominica',
 'Dharmasala', 'Dharamsala']

teams_ipl=['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Rajasthan Royals',
         'Chennai Super Kings',
         'Delhi Capitals',
         'Gujarat Titans',
         'Lucknow Supergiants']
teams_t20=[
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka'
]





# sidebar for navigation
with st.sidebar:
    selected = option_menu('Cricket Score Prediction System',

                           ['ODI SCORE',
                            'T20 SCORE',
                            'IPL SCORE'],
                           icons=['fan', 'fan', 'fan'],
                           default_index=1)


#ODI SCORE
if (selected == 'ODI SCORE'):

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Select batting team', sorted(teams_odi))
    with col2:
        bowling_team = st.selectbox('Select bowling team', sorted(teams_odi))

    city = st.selectbox('Select city', sorted(cities_odi))

    col3, col4, col5 = st.columns(3)

    with col3:
        current_score = st.number_input('Current Score')
    with col4:
        overs = st.number_input('Overs Completed (works for over>10)')
    with col5:
        wickets = st.number_input('Wickets left')

    last_ten = st.number_input('Runs scored in last 10 overs')

    if st.button('Predict Score'):
        balls_left = 300 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        input_df = pd.DataFrame(
            {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': city,
             'current_score': [current_score], 'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr],
             'last_ten': [last_ten]})
        result = odi_model.predict(input_df)
        st.header("Predicted Score - " + str(int(result[0])))


#T20 SCORE
if (selected == 'T20 SCORE'):

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Select batting team', sorted(teams_t20))
    with col2:
        bowling_team = st.selectbox('Select bowling team', sorted(teams_t20))

    city = st.selectbox('Select city', sorted(cities_t20))

    col3, col4, col5 = st.columns(3)

    with col3:
        current_score = st.number_input('Current Score')
    with col4:
        overs = st.number_input('Overs Completed (works for over>5)')
    with col5:
        wickets = st.number_input('Wickets left')

    last_five=st.number_input('Runs scored in last 5 overs')

    if st.button('Predict Score'):
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        input_df = pd.DataFrame(
            {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': city,
             'current_score': [current_score], 'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr],
             'last_five': [last_five]})
        result = t20_model.predict(input_df)
        st.header("Predicted Score - " + str(int(result[0])))



#IPL SCORE
if (selected == 'IPL SCORE'):

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Select batting team', sorted(teams_ipl))
    with col2:
        bowling_team = st.selectbox('Select bowling team', sorted(teams_ipl))

    city = st.selectbox('Select city', sorted(cities_ipl))

    col3, col4, col5 = st.columns(3)

    with col3:
        current_score = st.number_input('Current Score')
    with col4:
        overs = st.number_input('Overs Completed (works for over>5)')
    with col5:
        wickets = st.number_input('Wickets left')

    last_five=st.number_input('Runs scored in last 5 overs')

    if st.button('Predict Score'):
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        input_df = pd.DataFrame(
            {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': city,
             'current_score': [current_score], 'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr],
             'last_five': [last_five]})
        result =ipl_model.predict(input_df)
        st.header("Predicted Score - " + str(int(result[0])))