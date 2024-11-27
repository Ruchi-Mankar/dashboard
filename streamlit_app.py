import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
# import matplotlib.pyplot as plt
import altair as alt
# from prophet import Prophet


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='VJTI Canteen dashboard',
    page_icon=':bento:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

# gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :bento: VJTI Canteen dashboard

Made by: \n
Ruchi Mankar (211081028) \n
Nikhita Gharpure (211081022)
'''

# Add some spacing
''
''

# min_value = gdp_df['Year'].min()
# max_value = gdp_df['Year'].max()

# from_year, to_year = st.slider(
#     'Which years are you interested in?',
#     min_value=min_value,
#     max_value=max_value,
#     value=[min_value, max_value])

# countries = gdp_df['Country Code'].unique()

# if not len(countries):
#     st.warning("Select at least one country")

# selected_countries = st.multiselect(
#     'Which countries would you like to view?',
#     countries,
#     ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

# ''
# ''
# ''

# # Filter the data
# filtered_gdp_df = gdp_df[
#     (gdp_df['Country Code'].isin(selected_countries))
#     & (gdp_df['Year'] <= to_year)
#     & (from_year <= gdp_df['Year'])
# ]

# st.header('GDP over time', divider='gray')

# ''

# st.line_chart(
#     filtered_gdp_df,
#     x='Year',
#     y='GDP',
#     color='Country Code',
# )

# ''
# ''


# first_year = gdp_df[gdp_df['Year'] == from_year]
# last_year = gdp_df[gdp_df['Year'] == to_year]

# st.header(f'GDP in {to_year}', divider='gray')

# ''

# cols = st.columns(4)

# for i, country in enumerate(selected_countries):
#     col = cols[i % len(cols)]

#     with col:
#         first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
#         last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

#         if math.isnan(first_gdp):
#             growth = 'n/a'
#             delta_color = 'off'
#         else:
#             growth = f'{last_gdp / first_gdp:,.2f}x'
#             delta_color = 'normal'

#         st.metric(
#             label=f'{country} GDP',
#             value=f'{last_gdp:,.0f}B',
#             delta=growth,
#             delta_color=delta_color
#         )


menu_items_df = pd.read_csv('menu_items.csv')
holidays_df = pd.read_csv('holidays.csv')
ingredients_df = pd.read_csv('ingredients.csv')
sales_df = pd.read_csv('sales.csv')

# dates to yyyy-mm-dd format
sales_df['date'] = pd.to_datetime(sales_df['date'], format='%Y-%m-%d')  # Modify format as needed
holidays_df['start_date'] = pd.to_datetime(holidays_df['start_date'], format='%b %d, %Y')
holidays_df['end_date'] = pd.to_datetime(holidays_df['end_date'], format='%b %d, %Y')
#----------------------
sales_df['date'] = pd.to_datetime(sales_df['date'])
sales_df['day_of_week'] = sales_df['date'].dt.dayofweek
# Group sales data by 'day_of_week' and sum only the numeric columns
sales_by_day = sales_df.groupby('day_of_week').sum(numeric_only=True)
# Plot sales by day of the week for selected items (e.g., 'Tea' and 'Coffee')
sales_by_day.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
# Select relevant columns
chart_data = sales_by_day[['Vada Pav', 'Misal Pav', 'Samosa', 'Ragada Samosa']]
st.bar_chart(chart_data)
#----------------------

# # Load data
# sales_df = pd.read_csv('sales.csv')
# sales_df['date'] = pd.to_datetime(sales_df['date'])

# # Group by date and sum all sales
# daily_sales = sales_df.groupby('date').sum(numeric_only=True).reset_index()
# daily_sales = daily_sales[['date', 'Tea', 'Coffee', 'Cold Drinks', 'Vada Pav', 'Misal Pav', 'Samosa']]
# daily_sales['total_sales'] = daily_sales.sum(axis=1)

# # Prophet expects columns 'ds' and 'y'
# sales_for_prophet = daily_sales[['date', 'total_sales']].rename(columns={'date': 'ds', 'total_sales': 'y'})

# # Train a Prophet model
# model = Prophet()
# model.fit(sales_for_prophet)

# # User input for prediction days
# st.sidebar.header("Prediction Settings")
# x_days = st.sidebar.slider('Select number of days to predict', min_value=7, max_value=120, value=30, step=1)

# # Make future dataframe and predictions
# future = model.make_future_dataframe(periods=x_days)
# forecast = model.predict(future)

# # Display results
# st.title("Canteen Sales Forecast")
# st.write(f"Predictions for the next **{x_days} days**.")

# # Combine actual and predicted data for visualization
# forecast_chart_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales'})
# actual_chart_data = sales_for_prophet.rename(columns={'ds': 'Date', 'y': 'Actual Sales'})

# # Merge actual and predicted data for a combined chart
# chart_data = pd.merge(actual_chart_data, forecast_chart_data, on='Date', how='outer')

# # Line chart using Altair
# chart = alt.Chart(chart_data).transform_fold(
#     ['Actual Sales', 'Predicted Sales'],
#     as_=['Type', 'Sales']
# ).mark_line().encode(
#     x='Date:T',
#     y='Sales:Q',
#     color='Type:N',
#     tooltip=['Date:T', 'Sales:Q', 'Type:N']
# ).properties(
#     width=800,
#     height=400,
#     title="Actual vs Predicted Sales"
# )

# st.altair_chart(chart, use_container_width=True)

# # Show the prediction dataframe
# st.subheader("Forecast Data")
# st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(x_days))
