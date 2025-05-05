import pandas as pd

def load_and_clean_data(filepath: str, countries: list = None) -> pd.DataFrame:
    """
    Load and clean COVID-19 data for specified countries.
    
    Args:
        filepath (str): Path to the raw CSV file.
        countries (list): List of countries to include. Defaults to ['Kenya', 'United States', 'India'].
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with essential metrics.
    """
    if countries is None:
        countries = ['Kenya', 'United States', 'India', 'World']
    
    df = pd.read_csv(filepath)
    
    # Filter countries and handle dates
    df = df[df['location'].isin(countries)].copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['location', 'date'], inplace=True)
    
    # Calculate key metrics
    df['death_rate'] = (df['total_deaths'] / df['total_cases']) * 100
    df['vaccinated_percent'] = (df['total_vaccinations'] / df['population']) * 100
    
    # Forward-fill missing vaccinations
    df['total_vaccinations'] = df.groupby('location')['total_vaccinations'].ffill()
    
    # Select relevant columns
    keep_cols = [
        'date', 'location', 'total_cases', 'new_cases',
        'total_deaths', 'new_deaths', 'population',
        'death_rate', 'vaccinated_percent'
    ]
    
    return df[keep_cols].reset_index(drop=True)
