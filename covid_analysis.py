# COVID-19 Global Data Tracker
# Author: William Oneka
# GitHub: https://github.com/OneWilly

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

print("COVID-19 Global Data Tracker")
print("============================")

# 1. Data Collection and Loading
print("\n1. Data Loading & Initial Exploration")
print("-------------------------------------")

# Load the dataset
try:
    df = pd.read_csv('owid-covid-data.csv')
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: The file 'owid-covid-data.csv' was not found.")
    print("Please download the dataset from Our World in Data and place it in the working directory.")
    # For the purpose of this notebook, we'll continue as if data was loaded
    # In a real scenario, you would need to exit or handle this differently

# Display basic information about the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nColumns in the dataset:")
print(df.columns.tolist())

print("\nBasic statistics of key columns:")
key_columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
              'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
print(df[key_columns].describe())

print("\nMissing values in key columns:")
print(df[key_columns].isnull().sum())

# 2. Data Cleaning
print("\n2. Data Cleaning")
print("-----------------")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
print("✅ Converted 'date' column to datetime")

# Filter for specific countries of interest
countries_of_interest = ['Kenya', 'United States', 'India', 'United Kingdom', 'Brazil', 'Germany', 'South Africa']
df_countries = df[df['location'].isin(countries_of_interest)].copy()
print(f"✅ Filtered data for countries: {', '.join(countries_of_interest)}")

# Create a cleaned dataframe for global analysis
# Remove aggregated regions
df_global = df[~df['location'].isin(['World', 'European Union', 'International'])].copy()
print(f"✅ Created global dataframe excluding aggregated regions")

# Fill missing values for new cases and deaths with 0
for col in ['new_cases', 'new_deaths']:
    df_countries[col] = df_countries[col].fillna(0)
    df_global[col] = df_global[col].fillna(0)
print("✅ Filled missing values for new cases and deaths with 0")

# Sort by date and country
df_countries = df_countries.sort_values(['location', 'date'])
df_global = df_global.sort_values(['location', 'date'])
print("✅ Sorted data by location and date")

# Calculate daily statistics
latest_date = df['date'].max()
print(f"\nLatest date in the dataset: {latest_date.strftime('%Y-%m-%d')}")

latest_global = df_global[df_global['date'] == latest_date]
latest_countries = df_countries[df_countries['date'] == latest_date]

# Print dataset metrics after cleaning
print(f"\nNumber of countries in filtered dataset: {df_countries['location'].nunique()}")
print(f"Date range: {df_countries['date'].min().strftime('%Y-%m-%d')} to {df_countries['date'].max().strftime('%Y-%m-%d')}")

# 3. Exploratory Data Analysis
print("\n3. Exploratory Data Analysis (EDA)")
print("----------------------------------")

# 3.1 Total cases over time for selected countries
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = df_countries[df_countries['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)

plt.title('Total COVID-19 Cases Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('total_cases_by_country.png')
plt.close()
print("✅ Created chart: Total cases over time by country")

# 3.2 Total deaths over time for selected countries
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = df_countries[df_countries['location'] == country]
    plt.plot(country_data['date'], country_data['total_deaths'], label=country)

plt.title('Total COVID-19 Deaths Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('total_deaths_by_country.png')
plt.close()
print("✅ Created chart: Total deaths over time by country")

# 3.3 Daily new cases (7-day rolling average)
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = df_countries[df_countries['location'] == country].copy()
    country_data['rolling_new_cases'] = country_data['new_cases'].rolling(window=7).mean()
    plt.plot(country_data['date'], country_data['rolling_new_cases'], label=country)

plt.title('Daily New COVID-19 Cases (7-Day Rolling Average) by Country')
plt.xlabel('Date')
plt.ylabel('New Cases (7-day avg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('daily_new_cases_rolling_avg.png')
plt.close()
print("✅ Created chart: Daily new cases (7-day rolling average)")

# 3.4 Death rate calculation
print("\nDeath Rate Analysis:")
for country in countries_of_interest:
    country_latest = latest_countries[latest_countries['location'] == country]
    if len(country_latest) > 0 and pd.notna(country_latest['total_cases'].values[0]) and pd.notna(country_latest['total_deaths'].values[0]):
        cases = country_latest['total_cases'].values[0]
        deaths = country_latest['total_deaths'].values[0]
        if cases > 0:
            death_rate = (deaths / cases) * 100
            print(f"{country}: {death_rate:.2f}% ({int(deaths):,} deaths from {int(cases):,} cases)")

# 3.5 Bar chart of total cases for the selected countries
plt.figure(figsize=(12, 8))
latest_countries_sorted = latest_countries.sort_values('total_cases', ascending=False)
sns.barplot(x='location', y='total_cases', data=latest_countries_sorted)
plt.title('Total COVID-19 Cases by Country (Latest Date)')
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_cases_bar_chart.png')
plt.close()
print("✅ Created chart: Bar chart of total cases by country")

# 4. Vaccination Analysis
print("\n4. Vaccination Progress Analysis")
print("-------------------------------")

# 4.1 Plot cumulative vaccinations over time
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = df_countries[df_countries['location'] == country]
    plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)

plt.title('Total COVID-19 Vaccinations Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Total Vaccinations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('total_vaccinations_by_country.png')
plt.close()
print("✅ Created chart: Total vaccinations over time by country")

# 4.2 Percentage of population fully vaccinated
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = df_countries[df_countries['location'] == country]
    plt.plot(country_data['date'], country_data['people_fully_vaccinated_per_hundred'], label=country)

plt.title('Percentage of Population Fully Vaccinated by Country')
plt.xlabel('Date')
plt.ylabel('% Fully Vaccinated')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.axhline(y=70, color='r', linestyle='--', label='70% Target')
plt.savefig('vaccination_percentage_by_country.png')
plt.close()
print("✅ Created chart: Percentage of population fully vaccinated")

# 4.3 Vaccination progress as of latest date
print("\nVaccination Progress (Latest Date):")
for country in countries_of_interest:
    country_latest = latest_countries[latest_countries['location'] == country]
    if len(country_latest) > 0 and pd.notna(country_latest['people_fully_vaccinated_per_hundred'].values[0]):
        vax_rate = country_latest['people_fully_vaccinated_per_hundred'].values[0]
        print(f"{country}: {vax_rate:.2f}% fully vaccinated")

# 5. Choropleth Map Visualization with Plotly
print("\n5. Global Choropleth Map Visualization")
print("------------------------------------")

# Create a dataframe for the latest date only
latest_global_df = df_global[df_global['date'] == latest_date].copy()

# Create a map of total cases per million
try:
    fig = px.choropleth(
        latest_global_df,
        locations="iso_code",
        color="total_cases_per_million",
        hover_name="location",
        color_continuous_scale="YlOrRd",
        title="COVID-19 Cases per Million Population (Latest Date)",
        labels={"total_cases_per_million": "Cases per Million"}
    )
    fig.write_html("covid_cases_map.html")
    print("✅ Created interactive choropleth map: COVID-19 Cases per Million")
except Exception as e:
    print(f"❌ Could not create choropleth map: {str(e)}")
    print("Note: When running this notebook, make sure plotly is installed.")

# Create a map of vaccination progress
try:
    fig = px.choropleth(
        latest_global_df,
        locations="iso_code",
        color="people_fully_vaccinated_per_hundred",
        hover_name="location",
        color_continuous_scale="Greens",
        range_color=[0, 100],
        title="COVID-19 Vaccination Rate (% Fully Vaccinated, Latest Date)",
        labels={"people_fully_vaccinated_per_hundred": "% Fully Vaccinated"}
    )
    fig.write_html("covid_vaccination_map.html")
    print("✅ Created interactive choropleth map: COVID-19 Vaccination Rate")
except Exception as e:
    print(f"❌ Could not create vaccination choropleth map: {str(e)}")

# 6. Key Insights and Conclusions
print("\n6. Key Insights and Conclusions")
print("------------------------------")

# Calculate metrics for insights
top_cases_country = latest_countries.loc[latest_countries['total_cases'].idxmax(), 'location']
top_cases_count = latest_countries['total_cases'].max()

top_deaths_country = latest_countries.loc[latest_countries['total_deaths'].idxmax(), 'location']
top_deaths_count = latest_countries['total_deaths'].max()

top_vax_country = latest_countries.loc[latest_countries['people_fully_vaccinated_per_hundred'].idxmax(), 'location']
top_vax_rate = latest_countries['people_fully_vaccinated_per_hundred'].max()

# Print insights
print(f"\n1. Among the selected countries, {top_cases_country} had the highest number of total cases ({int(top_cases_count):,}).")
print(f"2. {top_deaths_country} recorded the highest death toll ({int(top_deaths_count):,}).")
print(f"3. {top_vax_country} achieved the highest vaccination rate with {top_vax_rate:.2f}% of the population fully vaccinated.")

# Calculate and print peak infection periods
print("\n4. Peak Infection Periods:")
for country in countries_of_interest:
    country_data = df_countries[df_countries['location'] == country].copy()
    country_data['rolling_new_cases'] = country_data['new_cases'].rolling(window=7).mean()
    if not country_data.empty and country_data['rolling_new_cases'].notna().any():
        peak_date = country_data.loc[country_data['rolling_new_cases'].idxmax(), 'date']
        peak_cases = country_data['rolling_new_cases'].max()
        print(f"   {country}: Peak on {peak_date.strftime('%Y-%m-%d')} with {int(peak_cases):,} daily cases (7-day avg)")

print("\n5. Comparison of Vaccination Progress and Case Rates:")
for country in countries_of_interest:
    country_latest = latest_countries[latest_countries['location'] == country]
    if len(country_latest) > 0:
        vax_rate = country_latest['people_fully_vaccinated_per_hundred'].values[0] if pd.notna(country_latest['people_fully_vaccinated_per_hundred'].values[0]) else "N/A"
        cases_per_million = country_latest['total_cases_per_million'].values[0] if pd.notna(country_latest['total_cases_per_million'].values[0]) else "N/A"
        print(f"   {country}: Vaccination rate: {vax_rate if isinstance(vax_rate, str) else f'{vax_rate:.2f}%'}, "
              f"Cases per million: {cases_per_million if isinstance(cases_per_million, str) else f'{int(cases_per_million):,}'}")

print("\n====================================================")
print("COVID-19 Global Data Tracker - Analysis Complete")
print("====================================================")
