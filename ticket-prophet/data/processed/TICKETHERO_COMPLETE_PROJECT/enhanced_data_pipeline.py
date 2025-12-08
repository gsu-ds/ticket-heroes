"""
Enhanced Data Cleaning and Panel Construction Pipeline
Integrates ALL data sources including GitHub, Last.fm, PHQ, and Setlist.fm
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataPipeline:
    def __init__(self, data_dir='/mnt/user-data/uploads'):
        self.data_dir = data_dir
        self.pollstar_years = [2022, 2023, 2024, 2025]
        
    def clean_github_data(self):
        """Clean GitHub 2016 concert data"""
        print("\nCleaning GitHub 2016 data...")
        
        github = pd.read_csv(f'{self.data_dir}/github_2016.csv')
        
        # Parse dates - handle timezone info
        github['date'] = pd.to_datetime(github['date'], errors='coerce', utc=True)
        github['date'] = github['date'].dt.tz_localize(None)  # Remove timezone
        github['year'] = github['date'].dt.year
        github['month'] = github['date'].dt.month
        github['dayofweek'] = github['date'].dt.dayofweek
        
        # Clean price data
        github['avg_price'] = (github['min_price'] + github['max_price']) / 2
        github['price_range'] = github['max_price'] - github['min_price']
        
        # Clean artist names
        github['artist'] = github['artist'].str.strip()
        
        # Clean num_years_active
        github['num_years_active'] = pd.to_numeric(
            github['num_years_active'], 
            errors='coerce'
        )
        
        # Create sold_out binary
        github['is_sold_out'] = (github['sold_out'] == 'True').astype(int)
        
        # Tickets per posting ratio
        github['tickets_per_posting'] = github['total_tickets'] / github['total_postings']
        
        print(f"  - GitHub records: {len(github)}")
        print(f"  - Unique artists: {github['artist'].nunique()}")
        print(f"  - Date range: {github['date'].min()} to {github['date'].max()}")
        
        return github
    
    def clean_lastfm_data(self):
        """Clean Last.fm artist data"""
        print("\nCleaning Last.fm data...")
        
        lastfm = pd.read_csv(f'{self.data_dir}/lastfm.csv')
        
        # Clean numeric columns (remove commas)
        lastfm['lastfm_listeners'] = pd.to_numeric(
            lastfm['lastfm_listeners'].str.replace(',', ''), 
            errors='coerce'
        )
        lastfm['lastfm_playcount'] = pd.to_numeric(
            lastfm['lastfm_playcount'].str.replace(',', ''), 
            errors='coerce'
        )
        
        # Clean artist names
        lastfm['primary_artist_name'] = lastfm['primary_artist_name'].str.strip()
        
        # Calculate playcount per listener
        lastfm['plays_per_listener'] = (
            lastfm['lastfm_playcount'] / lastfm['lastfm_listeners']
        )
        
        # Fill missing tags
        lastfm['lastfm_tag_1'] = lastfm['lastfm_tag_1'].fillna('Unknown')
        
        print(f"  - Last.fm records: {len(lastfm)}")
        print(f"  - Unique artists: {lastfm['primary_artist_name'].nunique()}")
        print(f"  - Records with data: {lastfm['lastfm_listeners'].notna().sum()}")
        
        return lastfm
    
    def clean_phq_data(self):
        """Clean PredictHQ event data"""
        print("\nCleaning PredictHQ data...")
        
        phq = pd.read_csv(f'{self.data_dir}/phq.csv')
        
        # Parse dates
        phq['start'] = pd.to_datetime(phq['start'], errors='coerce')
        phq['end'] = pd.to_datetime(phq['end'], errors='coerce')
        phq['updated'] = pd.to_datetime(phq['updated'], errors='coerce')
        
        # Extract temporal features
        phq['start_year'] = phq['start'].dt.year
        phq['start_month'] = phq['start'].dt.month
        phq['start_dayofweek'] = phq['start'].dt.dayofweek
        
        # Calculate duration if not provided
        phq['duration_calc'] = (phq['end'] - phq['start']).dt.total_seconds() / 3600
        phq['duration'] = phq['duration'].fillna(phq['duration_calc'])
        
        # Parse location (assuming format: "city, state")
        location_parts = phq['location'].str.split(',', expand=True, n=1)
        if location_parts.shape[1] >= 2:
            phq['phq_city'] = location_parts[0].str.strip()
            phq['phq_state'] = location_parts[1].str.strip()
        else:
            phq['phq_city'] = location_parts[0].str.strip()
            phq['phq_state'] = None
        
        # Clean attendance
        phq['phq_attendance'] = phq['phq_attendance'].fillna(
            phq.groupby('category')['phq_attendance'].transform('median')
        )
        
        print(f"  - PHQ records: {len(phq)}")
        print(f"  - Categories: {phq['category'].unique()}")
        print(f"  - Date range: {phq['start'].min()} to {phq['start'].max()}")
        
        return phq
    
    def clean_setlistfm_data(self):
        """Clean Setlist.fm concert data"""
        print("\nCleaning Setlist.fm data...")
        
        setlist = pd.read_csv(f'{self.data_dir}/setlistfm.csv')
        
        # Parse dates
        setlist['event_date'] = pd.to_datetime(setlist['event_date'], errors='coerce')
        setlist['event_last_updated'] = pd.to_datetime(
            setlist['event_last_updated'], 
            errors='coerce'
        )
        
        # Extract temporal features
        setlist['event_year'] = setlist['event_date'].dt.year
        setlist['event_month'] = setlist['event_date'].dt.month
        setlist['event_dayofweek'] = setlist['event_date'].dt.dayofweek
        
        # Clean artist names
        setlist['artist_name'] = setlist['artist_name'].str.strip()
        
        # Create has_tour indicator
        setlist['has_tour'] = setlist['tour_name'].notna().astype(int)
        
        # Filter to US only
        setlist_us = setlist[setlist['venue_country'] == 'United States'].copy()
        
        print(f"  - Setlist.fm records: {len(setlist)}")
        print(f"  - US records: {len(setlist_us)}")
        print(f"  - Unique artists: {setlist_us['artist_name'].nunique()}")
        print(f"  - Avg setlist length: {setlist_us['setlist_length'].mean():.1f}")
        
        return setlist_us
    
    def create_comprehensive_artist_dataset(self, spotify, lastfm, github, setlist):
        """Create comprehensive artist-level dataset merging all sources"""
        print("\nCreating comprehensive artist dataset...")
        
        # Start with Spotify artist stats
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
        
        # Merge with Last.fm (fuzzy matching by standardizing names)
        artist_stats['artist_name_std'] = artist_stats['artist_name'].str.lower().str.strip()
        lastfm['artist_name_std'] = lastfm['primary_artist_name'].str.lower().str.strip()
        
        artist_stats = artist_stats.merge(
            lastfm[['artist_name_std', 'lastfm_listeners', 'lastfm_playcount', 
                   'plays_per_listener', 'lastfm_tag_1']],
            on='artist_name_std',
            how='left'
        )
        
        # Merge with GitHub 2016 data (aggregate)
        github['artist_std'] = github['artist'].str.lower().str.strip()
        github_agg = github.groupby('artist_std').agg({
            'event_id': 'count',
            'avg_price': 'mean',
            'total_tickets': 'sum',
            'is_sold_out': 'mean',
            'num_years_active': 'first',
            'days_to_show': 'mean'
        }).reset_index()
        
        github_agg.columns = [
            'artist_name_std', 'github_events_2016', 'github_avg_price',
            'github_total_tickets', 'github_sellout_rate', 
            'num_years_active', 'github_avg_advance_days'
        ]
        
        artist_stats = artist_stats.merge(
            github_agg,
            on='artist_name_std',
            how='left'
        )
        
        # Merge with Setlist.fm data (aggregate)
        setlist['artist_std'] = setlist['artist_name'].str.lower().str.strip()
        setlist_agg = setlist.groupby('artist_std').agg({
            'event_id': 'count',
            'setlist_length': 'mean',
            'has_tour': 'mean',
            'venue_id': 'nunique'
        }).reset_index()
        
        setlist_agg.columns = [
            'artist_name_std', 'setlistfm_events', 'avg_setlist_length',
            'tour_frequency', 'unique_venues_setlist'
        ]
        
        artist_stats = artist_stats.merge(
            setlist_agg,
            on='artist_name_std',
            how='left'
        )
        
        # Clean up
        artist_stats = artist_stats.drop('artist_name_std', axis=1)
        
        print(f"  - Total artists: {len(artist_stats)}")
        print(f"  - With Last.fm data: {artist_stats['lastfm_listeners'].notna().sum()}")
        print(f"  - With GitHub data: {artist_stats['github_events_2016'].notna().sum()}")
        print(f"  - With Setlist.fm data: {artist_stats['setlistfm_events'].notna().sum()}")
        
        return artist_stats
    
    def create_event_level_dataset(self, tm, github, setlist, phq):
        """Create event-level dataset merging all sources"""
        print("\nCreating event-level dataset...")
        
        # Start with Ticketmaster
        events = tm.copy()
        events['data_source'] = 'Ticketmaster'
        
        # Add GitHub 2016 data
        github_events = github[[
            'event_id', 'date', 'artist', 'venue', 'city', 'state',
            'min_price', 'max_price', 'avg_price', 'total_tickets',
            'is_sold_out', 'year', 'month', 'dayofweek'
        ]].copy()
        
        github_events.columns = [
            'event_id', 'event_date', 'artist_name', 'venue_name', 
            'city', 'state_code', 'price_min', 'price_max', 'price_avg',
            'total_tickets', 'sold_out', 'year', 'month', 'dayofweek'
        ]
        github_events['data_source'] = 'GitHub'
        
        # Add Setlist.fm data
        setlist_events = setlist[[
            'event_id', 'event_date', 'artist_name', 'venue_name',
            'venue_city', 'venue_state', 'setlist_length', 'has_tour',
            'lat', 'lng', 'event_year', 'event_month', 'event_dayofweek'
        ]].copy()
        
        setlist_events.columns = [
            'event_id', 'event_date', 'artist_name', 'venue_name',
            'city', 'state_code', 'setlist_length', 'has_tour',
            'latitude', 'longitude', 'year', 'month', 'dayofweek'
        ]
        setlist_events['data_source'] = 'Setlist.fm'
        
        print(f"  - Ticketmaster events: {len(events)}")
        print(f"  - GitHub events: {len(github_events)}")
        print(f"  - Setlist.fm events: {len(setlist_events)}")
        print(f"  - Total events available: {len(events) + len(github_events) + len(setlist_events)}")
        
        return events, github_events, setlist_events
    
    def create_market_level_aggregations(self, events_tm, events_github, phq):
        """Create market-level aggregations from all event sources"""
        print("\nCreating market-level aggregations...")
        
        # Ticketmaster market aggregations (2025)
        tm_market = events_tm.groupby(['city', 'state_code']).agg({
            'event_id': 'count',
            'price_avg': 'mean',
            'artist_id': 'nunique',
            'venue_id': 'nunique'
        }).reset_index()
        
        tm_market.columns = [
            'city', 'state_code', 'tm_events_2025',
            'tm_avg_price_2025', 'tm_artists_2025', 'tm_venues_2025'
        ]
        tm_market['year'] = 2025
        
        # GitHub market aggregations (2016)
        github_market = events_github.groupby(['city', 'state_code']).agg({
            'event_id': 'count',
            'price_avg': 'mean',
            'artist_name': 'nunique',
            'total_tickets': 'sum',
            'sold_out': 'mean'
        }).reset_index()
        
        github_market.columns = [
            'city', 'state_code', 'github_events_2016',
            'github_avg_price_2016', 'github_artists_2016',
            'github_total_tickets_2016', 'github_sellout_rate_2016'
        ]
        github_market['year'] = 2016
        
        # PHQ market aggregations
        phq['phq_city'] = phq.get('phq_city', phq['location'])
        phq_market = phq.groupby(['phq_city']).agg({
            'id': 'count',
            'phq_attendance': 'sum',
            'rank': 'mean'
        }).reset_index()
        
        phq_market.columns = [
            'city', 'phq_events', 'phq_total_attendance', 'phq_avg_rank'
        ]
        
        print(f"  - TM market records: {len(tm_market)}")
        print(f"  - GitHub market records: {len(github_market)}")
        print(f"  - PHQ market records: {len(phq_market)}")
        
        return tm_market, github_market, phq_market
    
    def run_enhanced_pipeline(self, save_outputs=True):
        """Run the complete enhanced pipeline"""
        print("="*80)
        print("ENHANCED DATA CLEANING PIPELINE")
        print("="*80)
        
        # Load and clean existing data
        from data_cleaning_pipeline import DataCleaningPipeline
        base_pipeline = DataCleaningPipeline()
        
        print("\nLoading base datasets...")
        pollstar = base_pipeline.clean_pollstar_data()
        spotify, _ = base_pipeline.clean_spotify_data()
        tm, market_events_tm, artist_events_tm = base_pipeline.clean_ticketmaster_data()
        
        # Clean new datasets
        github = self.clean_github_data()
        lastfm = self.clean_lastfm_data()
        phq = self.clean_phq_data()
        setlist = self.clean_setlistfm_data()
        
        # Create comprehensive artist dataset
        comprehensive_artists = self.create_comprehensive_artist_dataset(
            spotify, lastfm, github, setlist
        )
        
        # Create event-level datasets
        events_tm, events_github, events_setlist = self.create_event_level_dataset(
            tm, github, setlist, phq
        )
        
        # Create market aggregations
        tm_market, github_market, phq_market = self.create_market_level_aggregations(
            events_tm, events_github, phq
        )
        
        # Save outputs
        if save_outputs:
            print("\n" + "="*80)
            print("SAVING ENHANCED DATA")
            print("="*80)
            
            outputs = {
                # Original cleaned data
                'github_clean.csv': github,
                'lastfm_clean.csv': lastfm,
                'phq_clean.csv': phq,
                'setlistfm_clean.csv': setlist,
                
                # Comprehensive datasets
                'comprehensive_artist_data.csv': comprehensive_artists,
                
                # Event-level data
                'events_github_2016.csv': events_github,
                'events_setlistfm.csv': events_setlist,
                
                # Market aggregations
                'market_tm_2025.csv': tm_market,
                'market_github_2016.csv': github_market,
                'market_phq.csv': phq_market,
            }
            
            for filename, df in outputs.items():
                filepath = f'/mnt/user-data/outputs/{filename}'
                df.to_csv(filepath, index=False)
                print(f"  ✓ Saved: {filename} ({len(df)} records, {len(df.columns)} columns)")
        
        print("\n" + "="*80)
        print("ENHANCED PIPELINE COMPLETE!")
        print("="*80)
        
        return {
            'github': github,
            'lastfm': lastfm,
            'phq': phq,
            'setlist': setlist,
            'comprehensive_artists': comprehensive_artists,
            'events_github': events_github,
            'events_setlist': events_setlist,
            'tm_market': tm_market,
            'github_market': github_market,
            'phq_market': phq_market
        }


if __name__ == "__main__":
    pipeline = EnhancedDataPipeline()
    results = pipeline.run_enhanced_pipeline(save_outputs=True)
    
    print("\n\nKEY INSIGHTS:")
    print(f"- Comprehensive artist data: {len(results['comprehensive_artists'])} artists")
    print(f"- Last.fm matches: {results['comprehensive_artists']['lastfm_listeners'].notna().sum()}")
    print(f"- GitHub 2016 matches: {results['comprehensive_artists']['github_events_2016'].notna().sum()}")
    print(f"- Setlist.fm matches: {results['comprehensive_artists']['setlistfm_events'].notna().sum()}")
