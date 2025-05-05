# COVID-19 Global Data Tracker

A comprehensive Python-based analysis tool for tracking and visualizing COVID-19 trends worldwide, including cases, deaths, and vaccination progress.

## Author
William Oneka
GitHub: [OneWilly](https://github.com/OneWilly)

## Project Overview
This project analyzes global COVID-19 data to identify trends, compare metrics across countries, and generate visual insights. The analysis covers:
- Case and death trends over time
- Vaccination rollout progress
- Country comparisons
- Interactive world maps

## Features
- Time series analysis of COVID-19 cases and deaths
- 7-day rolling averages for trend smoothing
- Death rate calculations
- Vaccination progress tracking
- Country comparison visualizations
- Interactive choropleth maps
- Data-driven key insights

## Data Sources
The project uses the Our World in Data COVID-19 dataset:
- [Our World in Data COVID-19 Dataset](https://github.com/owid/covid-19-data/tree/master/public/data)

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- jupyter (optional, for notebook version)

Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/OneWilly/covid19-data-tracker.git
   cd covid19-data-tracker
   ```

2. Download the dataset from Our World in Data and save as `owid-covid-data.csv` in the data directory

3. Run the analysis:
   ```
   python covid_analysis.py
   ```

4. View the generated visualizations in the outputs directory

## Generated Visualizations
- Total cases over time by country
- Total deaths over time by country
- Daily new cases (7-day rolling average)
- Vaccination progress comparison
- Choropleth maps of global case and vaccination data

## Key Insights
The analysis reveals critical patterns in the global COVID-19 landscape:
- Identification of countries with highest case loads
- Comparative analysis of death rates
- Vaccination progress and effectiveness
- Peak infection periods by country
- Relationship between vaccination rates and case numbers
