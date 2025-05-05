import pandas as pd

def clean_covid_data(filepath="data/owid-covid-data.csv", countries=None):
    """Load and clean COVID-19 data for specific countries."""
    if countries is None:
        countries = ['Kenya', 'United States', 'India', 'Brazil', 'Germany', 'World']
    
    df = pd.read_csv(filepath)
    df = df[df['location'].isin(countries)]
    
    # Convert date and handle missing values
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['location', 'date'], inplace=True)
    
    # Forward-fill vaccinations and calculate death rate
    df['total_vaccinations'] = df.groupby('location')['total_vaccinations'].ffill()
    df['death_rate'] = (df['total_deaths'] / df['total_cases']) * 100
    
    # Calculate new cases/deaths (if missing)
    df['new_cases'] = df.groupby('location')['total_cases'].diff().fillna(0)
    df['new_deaths'] = df.groupby('location')['total_deaths'].diff().fillna(0)
    
    # Filter columns
    keep_cols = ['date', 'location', 'total_cases', 'new_cases', 
                 'total_deaths', 'new_deaths', 'total_vaccinations', 
                 'population', 'death_rate']
    return df[keep_cols]
