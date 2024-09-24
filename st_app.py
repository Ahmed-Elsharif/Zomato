import streamlit as st
import pandas as pd
import pickle
import numpy as np

model_path = './pipeline_model.pkl'
csv_path = './df_encoded.csv'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv(csv_path)

# Function to initialize the input dictionary based on the dataset structure
def initialize_input(df):
    input_encoded = {col: 0 for col in df.columns if col != 'success'}  # Initialize all to 0, except the target
    return input_encoded

# Function to make predictions
def predict_success(input_data, model, df):
    # Initialize the input_encoded dictionary
    input_encoded = initialize_input(df)

    # Fill in the provided values from user inputs
    input_encoded['online_order'] = 1 if input_data['online_order'] == 'Yes' else 0
    input_encoded['book_table'] = 1 if input_data['book_table'] == 'Yes' else 0
    input_encoded['approx_cost(for two people)'] = input_data['approx_cost(for two people)']

    # Adding dish encodings
    dish_columns = [col for col in df.columns if col.startswith('dish_')]
    for dish in dish_columns:
        dish_name = dish.split('_')[1]
        input_encoded[dish] = 1 if dish_name in input_data['dishes'] else 0

    # Adding cuisine encodings
    cuisine_columns = [col for col in df.columns if col.startswith('cuisine_')]
    for cuisine in cuisine_columns:
        cuisine_name = cuisine.split('_')[1]
        input_encoded[cuisine] = 1 if cuisine_name in input_data['cuisines'] else 0

    # Adding location encodings
    location_columns = [col for col in df.columns if col.startswith('location_')]
    for location in location_columns:
        location_name = location.split('_')[1]
        input_encoded[location] = 1 if location_name == input_data['location'] else 0

    # Adding restaurant type encodings
    rest_type_columns = [col for col in df.columns if col.startswith('rest_type_')]
    for rest_type in rest_type_columns:
        rest_type_name = rest_type.split('_')[1]
        input_encoded[rest_type] = 1 if rest_type_name == input_data['rest_type'] else 0

    # Adding 'listed_in(type)' encoding
    listed_in_type_columns = [col for col in df.columns if col.startswith('listed_in(type)_')]
    for listed_in_type in listed_in_type_columns:
        listed_in_type_name = listed_in_type.split('_')[1]
        input_encoded[listed_in_type] = 1 if listed_in_type_name == input_data['listed_in(type)'] else 0

    # Adding listed_in(city) encodings (set to 1 if it matches the input city, else 0)
    listed_in_city_columns = [col for col in df.columns if col.startswith('listed_in(city)_')]
    for listed_in_city in listed_in_city_columns:
        listed_in_city_name = listed_in_city.split('_')[1]
        input_encoded[listed_in_city] = 1 if listed_in_city_name == input_data['listed_in(city)'] else 0   

    # Convert the input_encoded dictionary to a DataFrame for prediction
    input_df = pd.DataFrame([input_encoded])

    # Predict success using the trained pipeline model
    predicted_success = model.predict(input_df)
    return predicted_success[0]  # 1 = Success, 0 = Not Successful

# Streamlit UI for user inputs
st.title("Restaurant Success Prediction")

# Load data and model
df = load_data()
model = load_model()

# List of dishes for dropdown
dishes_list = ['dish_unknown', 'dish_pasta', 'dish_burgers', 'dish_cocktails', 'dish_pizza',
    'dish_biryani', 'dish_coffee']

# List of restaurant types for dropdown
rest_type_list = ['casual dining', 'cafe', 'quick bites', 'delivery', 'mess', 'dessert parlor', 'bakery', 
    'pub', 'takeaway', 'fine dining', 'beverage shop', 'sweet shop', 'bar', 'kiosk', 'food truck', 
    'microbrewery', 'lounge', 'food court', 'dhaba', 'club', 'irani cafee', 'confectionery', 
    'bhojanalya', 'meat shop']

# List of locations for dropdown
location_list = ['Banashankari', 'Basavanagudi', 'Mysore Road', 
    'Old Madras Road', 'Seshadripuram', 'Kammanahalli', 'Koramangala 6th Block', 
    'Yelahanka', 'Sahakara Nagar', 'Jalahalli', 'Hebbal', 'Nagarbhavi', 'Peenya', 'KR Puram']

# List of 'listed_in(type)' options for dropdown
listed_in_type_list = ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out', 'Drinks & nightlife', 'Pubs and bars']

# List of 'listed_in(city)' options for dropdown
listed_in_city_list = [
    'Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur', 'Brigade Road',
    'Brookefield', 'BTM', 'Church Street', 'Electronic City', 'Frazer Town', 'HSR',
    'Indiranagar']

# List of cuisines for dropdown
cuisines_list = ['North Indian','Chinese','South Indian','Fast Food','Continental','Biryani','Cafe',
 'Desserts','Beverages','Italian','Street Food','Bakery','Pizza','Burger','Seafood','Andhra','Ice Cream',
 'Mughlai']

# Create input form in the sidebar
st.sidebar.header("Enter Restaurant Features")

# User inputs
input_data = {
    'online_order': st.sidebar.selectbox('Online Order', ['Yes', 'No']),
    'book_table': st.sidebar.selectbox('Book Table', ['Yes', 'No']),
    'approx_cost(for two people)': st.sidebar.slider('Approximate Cost (for two people)', min_value=30, max_value=1100, step=10),
    'dishes': st.sidebar.multiselect('Select Dishes', dishes_list),
    'cuisines': st.sidebar.multiselect('Cuisines', cuisines_list),
    'location': st.sidebar.selectbox('Location', location_list),
    'rest_type': st.sidebar.selectbox('Restaurant Type', rest_type_list),
    'listed_in(type)': st.sidebar.selectbox('Listed In (Type)', listed_in_type_list),
    'listed_in(city)': st.sidebar.selectbox('Listed In (city)', listed_in_city_list)
}

# Prediction and result
if st.button('Predict Success'):
    predicted_success = predict_success(input_data, model, df)
    st.write(f"The predicted success is: {'Success' if predicted_success == 1 else 'Failure'}")

