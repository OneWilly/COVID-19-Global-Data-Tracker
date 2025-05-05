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

## Setup & Usage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/OneWilly/covid19-data-tracker.git
   cd covid19-data-tracker
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation
1. Download the dataset from [Our World in Data](https://github.com/owid/covid-19-data/tree/master/public/data)
2. Save as `owid-covid-data.csv` in the `data/` directory
3. If the data directory doesn't exist, create it:
   ```bash
   mkdir -p data
   ```

### Running the Analysis
1. Run the Python script:
   ```bash
   python covid_analysis.py
   ```

2. For notebook version (recommended for exploration):
   ```bash
   # Convert script to notebook
   python -c "import nbformat as nbf; code = open('covid_analysis.py').read(); nb = nbf.v4.new_notebook(); nb['cells'] = [nbf.v4.new_code_cell(code)]; nbf.write(nb, 'COVID-19-Analysis.ipynb')"
   
   # Run Jupyter
   jupyter notebook COVID-19-Analysis.ipynb
   ```

3. To generate PDF report from notebook:
   ```bash
   jupyter nbconvert --to pdf COVID-19-Analysis.ipynb
   ```

4. The visualizations will be generated in the root directory or `outputs/` if it exists

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
