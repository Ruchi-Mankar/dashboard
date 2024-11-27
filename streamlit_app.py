import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
import altair as alt
from prophet import Prophet
import matplotlib.pyplot as plt


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='VJTI Canteen dashboard',
    page_icon=':bento:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def Printhi():
    print('hi')
# Printhi()
# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :bento: VJTI Canteen dashboard

Made by: \n
Ruchi Mankar (211081028) \n
Nikhita Gharpure (211081022)
'''
menu_items_df = pd.read_csv('menu_items.csv')
holidays_df = pd.read_csv('holidays.csv')
ingredients_df = pd.read_csv('ingredients.csv')
sales_df = pd.read_csv('sales.csv')

# dates to yyyy-mm-dd format
sales_df['date'] = pd.to_datetime(sales_df['date'], format='%Y-%m-%d')  # Modify format as needed
holidays_df['start_date'] = pd.to_datetime(holidays_df['start_date'], format='%b %d, %Y')
holidays_df['end_date'] = pd.to_datetime(holidays_df['end_date'], format='%b %d, %Y')
#----------------------

# Load data
sales_df = pd.read_csv('sales.csv')
sales_df['date'] = pd.to_datetime(sales_df['date'])

# Group by date and sum all sales
daily_sales = sales_df.groupby('date').sum(numeric_only=True).reset_index()
daily_sales = daily_sales[['date','Water', 'Tea', 'Coffee', 'Cold Drinks', 'Dosa', 'Vada Pav', 'Misal Pav', 'Ragada Samosa', 'Samosa', 'Poha', 'Upma', 'Hakka Noodles', 'Fried Rice', 'Pav Bhaji', 'Vegetables', 'Chapati', 'Bread Pakoda', 'Sev Puri', 'Bhel Puri', 'Chinese Bhel', 'Sweet']]
# daily_sales = daily_sales[['date','Water']]
numeric_columns = daily_sales.select_dtypes(include=[np.number]).columns
daily_sales['total_sales'] = daily_sales[numeric_columns].sum(axis=1)
# daily_sales['total_sales'] = daily_sales.sum(axis=1)

# Prophet expects columns 'ds' and 'y'
sales_for_prophet = daily_sales[['date', 'total_sales']].rename(columns={'date': 'ds', 'total_sales': 'y'})

# Train a Prophet model
model = Prophet()
model.fit(sales_for_prophet)

# User input for prediction days
st.sidebar.header("Prediction Settings")
x_days = st.sidebar.slider('Select number of days to predict', min_value=7, max_value=120, value=30, step=1)

# Make future dataframe and predictions
future = model.make_future_dataframe(periods=x_days)
forecast = model.predict(future)

# Display results
st.title("Sales Forecast")
st.write(f"Predictions for the next **{x_days} days**.")

# Combine actual and predicted data for visualization
forecast_chart_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales'})
# actual_chart_data = sales_for_prophet.rename(columns={'ds': 'Date', 'y': 'Actual Sales'})

# Merge actual and predicted data for a combined chart
# chart_data = pd.merge(actual_chart_data, forecast_chart_data, on='Date', how='outer')

# Line chart using Altair
chart = alt.Chart(forecast_chart_data).mark_line().encode(
    x='Date:T',
    y='Predicted Sales:Q',
    tooltip=['Date:T', 'Predicted Sales:Q']
).properties(
    width=800,
    height=400,
    title="Forecasted Sales"
)

st.altair_chart(chart, use_container_width=True)

# Show the prediction dataframe
st.subheader("Forecast Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(x_days))
# st.write(forecast[['date', 'predicted value', 'lower bound', 'upper bound']].tail(x_days))
#----------------------------
# Load the sales data
sales_df = pd.read_csv('sales.csv')
sales_df['date'] = pd.to_datetime(sales_df['date'])

# Extract day of the week (0=Monday, 6=Sunday)
sales_df['day_of_week'] = sales_df['date'].dt.dayofweek

# Group by day_of_week and calculate average sales for each menu item
sales_by_day_avg = sales_df.groupby('day_of_week').mean(numeric_only=True)

# Rename days of the week for better readability
sales_by_day_avg.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# List of menu items
menu_items = ['Water', 'Tea', 'Coffee', 'Cold Drinks', 'Dosa', 'Vada Pav', 'Misal Pav', 'Ragada Samosa', 'Samosa', 'Poha', 'Upma', 'Hakka Noodles', 'Fried Rice', 'Pav Bhaji', 'Vegetables', 'Chapati', 'Bread Pakoda', 'Sev Puri', 'Bhel Puri', 'Chinese Bhel', 'Sweet']

# Add 'All' option to select all items
menu_items = ['All'] + menu_items
st.title("Average Sales by Day of the Week for Selected Menu Item")
# Dropdown for selecting menu item
selected_item = st.selectbox("Select Menu Item", menu_items)

# Show chart based on the selection
if selected_item == 'All':
    # Show average sales for all items
    chart_data_avg = sales_by_day_avg
else:
    # Show average sales for selected item
    chart_data_avg = sales_by_day_avg[[selected_item]]

# Create the bar chart
st.bar_chart(chart_data_avg)
#_______________________________________________
sales_df = pd.read_csv('sales.csv')
ingredients_df = pd.read_csv('ingredients.csv')

# Create menu items mapping (you can adjust based on your data)
menu_items_mapping = {
    'Water': 0,
    'Tea': 1,
    'Coffee': 2,
    'Cold Drinks': 3,
    'Dosa': 4,
    'Vada Pav': 5,
    'Samosa': 6,
    'Ragada Samosa': 7,
    'Misal Pav': 8,
    'Poha': 9,
    'Upma': 10,
    'Hakka Noodles': 11,
    'Fried Rice': 12,
    'Pav Bhaji': 13,
    'Vegetables': 14,
    'Chapati': 15,
    'Bread Pakoda': 16,
    'Sev Puri': 17,
    'Bhel Puri': 18,
    'Chinese Bhel': 19,
    'Sweet': 20
}

# Melt the sales dataframe to get it into a long format (date, item, sales)
sales_long_df = sales_df.melt(id_vars=["date", "day"], var_name="item", value_name="sales")

# Map each item to its corresponding meal_id using the 'menu_items_mapping'
sales_long_df['meal_id'] = sales_long_df['item'].map(menu_items_mapping)

# Calculate ingredient requirements
ingredient_requirements = []

for _, row in sales_long_df.iterrows():
    meal_id = row['meal_id']
    servings_sold = row['sales']  # number of servings sold for this menu item on the given day

    # Get the ingredients for this meal_id
    ingredients_for_meal = ingredients_df[ingredients_df['meal_id'] == meal_id]

    # For each ingredient, calculate the required quantity
    for _, ingredient_row in ingredients_for_meal.iterrows():
        ingredient = ingredient_row['ingredient']
        quantity_needed = ingredient_row['quantity_needed']
        unit = ingredient_row['unit']

        # Calculate the total required quantity for this ingredient
        total_quantity = quantity_needed * servings_sold

        # Append the results to the ingredient_requirements list
        ingredient_requirements.append({
            'ingredient': ingredient,
            'meal_id': meal_id,
            'item': row['item'],  # item name from sales data
            'sales': servings_sold,
            'quantity_needed': quantity_needed,
            'unit': unit,
            'total_quantity': total_quantity,
            'date': row['date']
        })

# Convert the results to a DataFrame
ingredient_requirements_df = pd.DataFrame(ingredient_requirements)

# Streamlit App
st.title("Canteen Ingredient Requirement Dashboard")

# Dropdown to select menu item or all
menu_item = st.selectbox("Select a Menu Item", options=["All"] + list(menu_items_mapping.keys()))
if menu_item != "All":
    filtered_data = ingredient_requirements_df[ingredient_requirements_df['item'] == menu_item]
else:
    filtered_data = ingredient_requirements_df

# Display the filtered ingredient requirements data
st.subheader("Ingredient Requirements")
st.write(filtered_data)

# Optionally, you can plot total ingredient requirements for a specific menu item
st.subheader("Total Ingredient Requirement")
ingredient_totals = filtered_data.groupby(['ingredient', 'unit'])['total_quantity'].sum().reset_index()

# Bar chart of ingredient totals
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(ingredient_totals['ingredient'], ingredient_totals['total_quantity'])
ax.set_xlabel("Ingredient")
ax.set_ylabel("Total Quantity")
ax.set_title("Total Ingredient Requirements")

# Display the plot
st.pyplot(fig)
#------- ------------- ---------------- ------------
# Load sales data (menu items sold)
sales_df = pd.read_csv('sales.csv')

# Load ingredients data (ingredients needed for each menu item)
ingredients_df = pd.read_csv('ingredients.csv')

# Create a mapping from menu item to meal_id
menu_items_mapping = {
    'Water': 0,
    'Tea': 1,
    'Coffee': 2,
    'Cold Drinks': 3,
    'Dosa': 4,
    'Vada Pav': 5,
    'Samosa': 6,
    'Ragada Samosa': 7,
    'Misal Pav': 8,
    'Poha': 9,
    'Upma': 10,
    'Hakka Noodles': 11,
    'Fried Rice': 12,
    'Pav Bhaji': 13,
    'Vegetables': 14,
    'Chapati': 15,
    'Bread Pakoda': 16,
    'Sev Puri': 17,
    'Bhel Puri': 18,
    'Chinese Bhel': 19,
    'Sweet': 20
}

# Melt the sales dataframe to get it into a long format (date, item, sales)
sales_long_df = sales_df.melt(id_vars=["date", "day"], var_name="item", value_name="sales")

# Map each item to its corresponding meal_id
sales_long_df['meal_id'] = sales_long_df['item'].map(menu_items_mapping)

# Now, for each item (meal_id), calculate the required ingredient quantities based on the sales
ingredient_requirements = []

for _, row in sales_long_df.iterrows():
    meal_id = row['meal_id']
    servings_sold = row['sales']  # number of servings sold for this menu item on the given day

    # Get the ingredients for this meal_id
    ingredients_for_meal = ingredients_df[ingredients_df['meal_id'] == meal_id]

    # For each ingredient, calculate the required quantity
    for _, ingredient_row in ingredients_for_meal.iterrows():
        ingredient = ingredient_row['ingredient']
        quantity_needed = ingredient_row['quantity_needed']
        unit = ingredient_row['unit']

        # Calculate the total required quantity for this ingredient
        total_quantity = quantity_needed * servings_sold

        # Append the results to the ingredient_requirements list
        ingredient_requirements.append({
            'ingredient': ingredient,
            'meal_id': meal_id,
            'item': row['item'],  # item name from sales data
            'sales': servings_sold,
            'quantity_needed': quantity_needed,
            'unit': unit,
            'total_quantity': total_quantity,
            'date': row['date']
        })

# Convert the results to a DataFrame
ingredient_requirements_df = pd.DataFrame(ingredient_requirements)

# Streamlit Dropdown to select ingredients
selected_ingredient = st.selectbox(
    'Select Ingredient', 
    ingredient_requirements_df['ingredient'].unique()
)

# Filter data based on selected ingredient
filtered_data = ingredient_requirements_df[ingredient_requirements_df['ingredient'] == selected_ingredient]

# Group by date and sum the total quantity required for the selected ingredient
ingredient_totals = filtered_data.groupby('date')['total_quantity'].sum().reset_index()

# Plot the data
plt.figure(figsize=(12, 6))

# Line plot for the selected ingredient over time
plt.plot(ingredient_totals['date'], ingredient_totals['total_quantity'], marker='o')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add titles and labels
plt.title(f"Total Quantity of {selected_ingredient} Required Over Time", fontsize=16)
plt.ylabel("Total Quantity", fontsize=12)

# Optional: Add gridlines for better readability
plt.grid(True)

# Show the plot
plt.tight_layout()  # Ensures everything fits without overlap
st.pyplot(plt)