"""
Data Cleaning and Panel Construction Pipeline
Creates a clean panel dataset for model testing from multiple data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaningPipeline:
    def __init__(self, data_dir='/mnt/user-data/uploads'):
        self.data_dir = data_dir
        self.pollstar_years = [2022, 2023, 2024, 2025]
        
    def clean_pollstar_data(self):
        """Clean and combine Pollstar market data across years"""
        print("Cleaning Pollstar data...")
        
        dfs = []
        for year in self.pollstar_years:
            df = pd.read_excel(f'{self.data_dir}/{year}pollstar.xlsx')
            dfs.append(df)
        
        # Combine all years
        pollstar = pd.concat(dfs, ignore_index=True)
        
        # Clean market names (standardize)
        pollstar['market'] = pollstar['market'].str.strip()
        
        # Handle missing avg_price_change (set to 0 for first year)
        pollstar['avg_price_change'] = pollstar['avg_price_change'].fillna(0)
        
        # Create additional features
        pollstar['revenue_per_show'] = pollstar['gross'] / pollstar['shows']
        pollstar['tickets_per_show'] = pollstar['tickets'] / pollstar['shows']
        
        # Sort by year and market
        pollstar = pollstar.sort_values(['market', 'year']).reset_index(drop=True)
        
        print(f"  - Pollstar records: {len(pollstar)}")
        print(f"  - Unique markets: {pollstar['market'].nunique()}")
        print(f"  - Years: {sorted(pollstar['year'].unique())}")
        
        return pollstar
    
    def clean_spotify_data(self):
        """Clean Spotify artist/track data"""
        print("\nCleaning Spotify data...")
        
        spotify = pd.read_csv(f'{self.data_dir}/spotify_kaggle.csv')
        
        # Drop duplicate column
        if 'track_duration_min' in spotify.columns:
            spotify = spotify.drop('track_duration_min', axis=1)
        
        # Calculate duration in minutes properly
        spotify['track_duration_min'] = spotify['track_duration_ms'] / 60000
        
        # Handle missing values
        spotify['artist_name'] = spotify['artist_name'].fillna('Unknown')
        spotify['track_name'] = spotify['track_name'].fillna('Unknown')
        spotify['album_name'] = spotify['album_name'].fillna('Unknown')
        
        # Fill missing numeric values with median
        spotify['artist_popularity'] = spotify['artist_popularity'].fillna(
            spotify['artist_popularity'].median()
        )
        spotify['artist_followers'] = spotify['artist_followers'].fillna(
            spotify['artist_followers'].median()
        )
        
        # Parse album release date
        spotify['album_release_year'] = pd.to_datetime(
            spotify['album_release_date'], 
            errors='coerce'
        ).dt.year
        
        # Create artist-level aggregations
        artist_stats = spotify.groupby('artist_name').agg({
            'artist_popularity': 'first',
            'artist_followers': 'first',
            'track_popularity': 'mean',
            'track_id': 'count',
            'explicit': 'mean',
            'track_duration_min': 'mean'
        }).reset_index()
        
        artist_stats.columns = [
            'artist_name', 'artist_popularity', 'artist_followers',
            'avg_track_popularity', 'total_tracks', 'explicit_ratio',
            'avg_track_duration'
        ]
        
        print(f"  - Spotify track records: {len(spotify)}")
        print(f"  - Unique artists: {spotify['artist_name'].nunique()}")
        print(f"  - Artist-level aggregations: {len(artist_stats)}")
        
        return spotify, artist_stats
    
    def clean_ticketmaster_data(self):
        """Clean Ticketmaster event data"""
        print("\nCleaning Ticketmaster data...")
        
        tm = pd.read_csv(f'{self.data_dir}/ticketmaster_complete.csv')
        
        # Parse dates
        tm['event_date'] = pd.to_datetime(tm['event_date'], errors='coerce')
        tm['onsale_date'] = pd.to_datetime(tm['onsale_date'], errors='coerce')
        
        # Extract temporal features
        tm['event_year'] = tm['event_date'].dt.year
        tm['event_month'] = tm['event_date'].dt.month
        tm['event_dayofweek'] = tm['event_date'].dt.dayofweek
        tm['event_quarter'] = tm['event_date'].dt.quarter
        
        # Calculate advance sale period
        tm['advance_days'] = (tm['event_date'] - tm['onsale_date']).dt.days
        
        # Handle price data
        tm['price_range'] = tm['price_max'] - tm['price_min']
        tm['price_avg'] = (tm['price_max'] + tm['price_min']) / 2
        
        # Fill missing prices with median by genre
        for col in ['price_min', 'price_max', 'price_avg']:
            tm[col] = tm.groupby('genre')[col].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Clean artist names
        tm['artist_name'] = tm['artist_name'].fillna('Unknown')
        tm['artist_name'] = tm['artist_name'].str.strip()
        
        # Filter to valid years
        tm = tm[tm['event_year'].between(2022, 2025)]
        
        # Create market-level aggregations
        market_events = tm.groupby(['city', 'state_code', 'event_year']).agg({
            'event_id': 'count',
            'price_avg': 'mean',
            'artist_id': 'nunique'
        }).reset_index()
        
        market_events.columns = [
            'city', 'state_code', 'year', 'num_events',
            'avg_ticket_price', 'unique_artists'
        ]
        
        # Create artist-level aggregations
        artist_events = tm.groupby(['artist_name', 'event_year']).agg({
            'event_id': 'count',
            'price_avg': 'mean',
            'venue_id': 'nunique',
            'city': 'nunique'
        }).reset_index()
        
        artist_events.columns = [
            'artist_name', 'year', 'num_events',
            'avg_ticket_price', 'unique_venues', 'unique_cities'
        ]
        
        print(f"  - Ticketmaster events: {len(tm)}")
        print(f"  - Date range: {tm['event_date'].min()} to {tm['event_date'].max()}")
        print(f"  - Unique artists: {tm['artist_name'].nunique()}")
        print(f"  - Market-year combinations: {len(market_events)}")
        
        return tm, market_events, artist_events
    
    def create_market_panel(self, pollstar, market_events):
        """Create market-level panel data"""
        print("\nCreating market-level panel...")
        
        # Map city names to markets (simplified mapping)
        city_to_market = {
            'New York': 'New York',
            'Los Angeles': 'Los Angeles',
            'Las Vegas': 'Las Vegas',
            'Chicago': 'Chicago',
            'San Francisco': 'San Francisco',
            'Boston': 'Boston',
            'Philadelphia': 'Philadelphia',
            'Dallas': 'Dallas',
            'Houston': 'Houston',
            'Atlanta': 'Atlanta',
            'Miami': 'Miami',
            'Seattle': 'Seattle',
            'Nashville': 'Nashville'
        }
        
        market_events['market'] = market_events['city'].map(city_to_market)
        
        # Aggregate to market level
        tm_market = market_events.groupby(['market', 'year']).agg({
            'num_events': 'sum',
            'avg_ticket_price': 'mean',
            'unique_artists': 'sum'
        }).reset_index()
        
        tm_market.columns = [
            'market', 'year', 'tm_events',
            'tm_avg_price', 'tm_unique_artists'
        ]
        
        # Merge with Pollstar data
        panel = pollstar.merge(
            tm_market,
            on=['market', 'year'],
            how='left'
        )
        
        # Create lagged variables
        panel = panel.sort_values(['market', 'year'])
        for col in ['gross', 'tickets', 'avg_price', 'shows']:
            panel[f'{col}_lag1'] = panel.groupby('market')[col].shift(1)
        
        # Create growth rates
        panel['gross_growth'] = panel.groupby('market')['gross'].pct_change()
        panel['ticket_growth'] = panel.groupby('market')['tickets'].pct_change()
        
        print(f"  - Market panel records: {len(panel)}")
        print(f"  - Markets with complete data: {panel.groupby('market').size().min()}")
        
        return panel
    
    def create_artist_panel(self, artist_stats, artist_events):
        """Create artist-level panel data"""
        print("\nCreating artist-level panel...")
        
        # Merge Spotify and Ticketmaster artist data
        artist_panel = artist_events.merge(
            artist_stats,
            on='artist_name',
            how='left'
        )
        
        # Create additional features
        artist_panel['events_per_venue'] = (
            artist_panel['num_events'] / artist_panel['unique_venues']
        )
        
        # Sort by artist and year
        artist_panel = artist_panel.sort_values(['artist_name', 'year'])
        
        # Create lagged variables
        for col in ['num_events', 'avg_ticket_price']:
            artist_panel[f'{col}_lag1'] = artist_panel.groupby('artist_name')[col].shift(1)
        
        print(f"  - Artist panel records: {len(artist_panel)}")
        print(f"  - Unique artists: {artist_panel['artist_name'].nunique()}")
        
        return artist_panel
    
    def generate_summary_stats(self, df, name):
        """Generate summary statistics for a dataframe"""
        print(f"\n{'='*60}")
        print(f"Summary Statistics: {name}")
        print('='*60)
        
        # Numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = df[numeric_cols].describe()
        print(summary)
        
        # Missing values
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            print(missing)
        else:
            print("No missing values!")
        
        return summary
    
    def run_pipeline(self, save_outputs=True):
        """Run the complete data cleaning pipeline"""
        print("="*80)
        print("DATA CLEANING PIPELINE")
        print("="*80)
        
        # Clean individual datasets
        pollstar = self.clean_pollstar_data()
        spotify, artist_stats = self.clean_spotify_data()
        tm, market_events, artist_events = self.clean_ticketmaster_data()
        
        # Create panel datasets
        market_panel = self.create_market_panel(pollstar, market_events)
        artist_panel = self.create_artist_panel(artist_stats, artist_events)
        
        # Generate summaries
        self.generate_summary_stats(market_panel, "Market Panel")
        self.generate_summary_stats(artist_panel, "Artist Panel")
        
        # Save outputs
        if save_outputs:
            print("\n" + "="*80)
            print("SAVING CLEANED DATA")
            print("="*80)
            
            outputs = {
                'market_panel.csv': market_panel,
                'artist_panel.csv': artist_panel,
                'pollstar_clean.csv': pollstar,
                'spotify_clean.csv': spotify,
                'artist_stats.csv': artist_stats,
                'ticketmaster_clean.csv': tm,
                'market_events.csv': market_events,
                'artist_events.csv': artist_events
            }
            
            for filename, df in outputs.items():
                filepath = f'/mnt/user-data/outputs/{filename}'
                df.to_csv(filepath, index=False)
                print(f"  ✓ Saved: {filename} ({len(df)} records, {len(df.columns)} columns)")
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        
        return {
            'market_panel': market_panel,
            'artist_panel': artist_panel,
            'pollstar': pollstar,
            'spotify': spotify,
            'artist_stats': artist_stats,
            'ticketmaster': tm,
            'market_events': market_events,
            'artist_events': artist_events
        }


if __name__ == "__main__":
    # Run the pipeline
    pipeline = DataCleaningPipeline()
    results = pipeline.run_pipeline(save_outputs=True)
    
    print("\n\nDATA PANEL STRUCTURE:")
    print("\nMarket Panel Columns:")
    print(results['market_panel'].columns.tolist())
    print("\nArtist Panel Columns:")
    print(results['artist_panel'].columns.tolist())
