"""
FINAL DATA INTEGRATION - DMA Strength + Secondary Market Data
Integrates market strength metrics and event-level secondary market features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinalDataIntegration:
    def __init__(self):
        self.output_dir = '/mnt/user-data/outputs'
        self.temp_dir = '/home/claude'
        
    def load_existing_data(self):
        """Load all existing cleaned datasets"""
        print("Loading existing data...")
        
        self.market_panel = pd.read_csv(f'{self.output_dir}/market_panel.csv')
        self.artist_panel = pd.read_csv(f'{self.output_dir}/artist_panel.csv')
        self.comprehensive_artists = pd.read_csv(f'{self.output_dir}/comprehensive_artist_data.csv')
        self.pollstar = pd.read_csv(f'{self.output_dir}/pollstar_clean.csv')
        self.ticketmaster = pd.read_csv(f'{self.output_dir}/ticketmaster_clean.csv')
        
        print(f"  ✓ Market panel: {self.market_panel.shape}")
        print(f"  ✓ Artist panel: {self.artist_panel.shape}")
        print(f"  ✓ Comprehensive artists: {self.comprehensive_artists.shape}")
        
    def load_new_data(self):
        """Load DMA strength and secondary market data"""
        print("\nLoading NEW data sources...")
        
        # DMA market strength
        self.dma_strength = pd.read_csv(f'{self.temp_dir}/dma_market_strength.csv')
        print(f"  ✓ DMA strength: {self.dma_strength.shape}")
        
        # Secondary market event features
        self.seat_features = pd.read_csv('/mnt/user-data/uploads/seatdata_clean_with_features.csv')
        self.seat_ts = pd.read_csv('/mnt/user-data/uploads/seatdata_ts_features.csv')
        print(f"  ✓ Secondary market features: {self.seat_features.shape}")
        print(f"  ✓ Time series features: {self.seat_ts.shape}")
        
    def standardize_market_names(self, df, market_col='market'):
        """Standardize market names for merging"""
        df = df.copy()
        df[f'{market_col}_std'] = (df[market_col]
                                    .str.strip()
                                    .str.replace('  ', ' ')
                                    .str.replace('(', ' (')
                                    .str.replace(')', ') ')
                                    .str.strip())
        return df
    
    def integrate_dma_strength_to_market_panel(self):
        """Add DMA strength metrics to market panel"""
        print("\nIntegrating DMA strength into market panel...")
        
        # Standardize market names in both datasets
        market_panel = self.standardize_market_names(self.market_panel, 'market')
        dma_strength = self.standardize_market_names(self.dma_strength, 'market')
        
        # Merge
        enhanced_market_panel = market_panel.merge(
            dma_strength[['market_std', 'dma_static_strength', 'annual_sales_avg', 
                         'avg_ticket_price_dma', 'annual_sales_current']],
            on='market_std',
            how='left'
        )
        
        # Calculate additional DMA-based features
        enhanced_market_panel['strength_vs_avg'] = (
            enhanced_market_panel['dma_static_strength'] / 
            enhanced_market_panel['dma_static_strength'].mean()
        )
        
        enhanced_market_panel['price_premium_vs_dma'] = (
            enhanced_market_panel['avg_price'] / 
            enhanced_market_panel['avg_ticket_price_dma']
        )
        
        enhanced_market_panel['sales_momentum'] = (
            enhanced_market_panel['annual_sales_current'] / 
            enhanced_market_panel['annual_sales_avg']
        )
        
        # Drop temporary columns
        enhanced_market_panel = enhanced_market_panel.drop('market_std', axis=1)
        
        matches = enhanced_market_panel['dma_static_strength'].notna().sum()
        print(f"  ✓ Matched {matches}/{len(enhanced_market_panel)} markets")
        print(f"  ✓ Added 8 new variables")
        
        return enhanced_market_panel
    
    def create_secondary_market_dataset(self):
        """Create event-level dataset with secondary market features"""
        print("\nCreating secondary market event dataset...")
        
        # Use the more complete dataset
        events = self.seat_features.copy()
        
        # Parse venue location to extract city/state
        venue_split = events['venue_location'].str.split(',', expand=True, n=1)
        if venue_split.shape[1] >= 2:
            events['city'] = venue_split[0].str.strip()
            events['state'] = venue_split[1].str.strip()
        elif venue_split.shape[1] == 1:
            events['city'] = venue_split[0].str.strip()
            events['state'] = None
        else:
            events['city'] = None
            events['state'] = None
        
        # Parse event date
        events['event_date'] = pd.to_datetime(events['event_date_time'], errors='coerce')
        events['event_year'] = events['event_date'].dt.year
        events['event_month'] = events['event_date'].dt.month
        events['event_dayofweek'] = events['event_date'].dt.dayofweek
        
        # Clean artist names
        events['artist_name'] = events['artist_name'].str.strip()
        
        # Create demand indicators
        events['high_demand'] = (events['active_listings_count'] > 
                                events['active_listings_count'].median()).astype(int)
        
        events['high_volatility'] = (events['price_volatility'] > 
                                    events['price_volatility'].median()).astype(int)
        
        events['premium_pricing'] = (events['median_secondary_price'] > 
                                    events['median_secondary_price'].median()).astype(int)
        
        print(f"  ✓ Secondary market events: {len(events)}")
        print(f"  ✓ Unique artists: {events['artist_name'].nunique()}")
        print(f"  ✓ Date range: {events['event_date'].min()} to {events['event_date'].max()}")
        
        return events
    
    def aggregate_secondary_market_by_artist(self, events):
        """Aggregate secondary market metrics by artist"""
        print("\nAggregating secondary market data by artist...")
        
        artist_secondary = events.groupby('artist_name').agg({
            'event_id_clean': 'count',
            'median_secondary_price': 'mean',
            'price_volatility': 'mean',
            'price_range': 'mean',
            'active_listings_count': 'mean',
            'avg_daily_price_change': 'mean',
            'high_demand': 'mean',
            'high_volatility': 'mean',
            'premium_pricing': 'mean',
            'valid_price_observations': 'mean'
        }).reset_index()
        
        artist_secondary.columns = [
            'artist_name', 'secondary_market_events', 'avg_secondary_price',
            'avg_price_volatility', 'avg_price_range', 'avg_active_listings',
            'avg_daily_price_change', 'high_demand_rate', 'high_volatility_rate',
            'premium_pricing_rate', 'avg_price_observations'
        ]
        
        print(f"  ✓ Artists with secondary market data: {len(artist_secondary)}")
        
        return artist_secondary
    
    def enhance_comprehensive_artists(self, artist_secondary):
        """Add secondary market features to comprehensive artist dataset"""
        print("\nEnhancing comprehensive artist dataset...")
        
        # Standardize artist names
        comp_artists = self.comprehensive_artists.copy()
        comp_artists['artist_name_std'] = comp_artists['artist_name'].str.lower().str.strip()
        
        artist_secondary['artist_name_std'] = artist_secondary['artist_name'].str.lower().str.strip()
        
        # Merge
        enhanced_artists = comp_artists.merge(
            artist_secondary.drop('artist_name', axis=1),
            on='artist_name_std',
            how='left'
        )
        
        # Drop temporary column
        enhanced_artists = enhanced_artists.drop('artist_name_std', axis=1)
        
        matches = enhanced_artists['secondary_market_events'].notna().sum()
        print(f"  ✓ Matched {matches}/{len(enhanced_artists)} artists with secondary market data")
        print(f"  ✓ Added 10 new variables")
        
        return enhanced_artists
    
    def create_market_secondary_aggregations(self, events):
        """Create market-level aggregations from secondary market data"""
        print("\nCreating market-level secondary market aggregations...")
        
        # Aggregate by city
        market_secondary = events.groupby('city').agg({
            'event_id_clean': 'count',
            'median_secondary_price': 'mean',
            'price_volatility': 'mean',
            'active_listings_count': 'mean',
            'high_demand': 'mean',
            'artist_name': 'nunique'
        }).reset_index()
        
        market_secondary.columns = [
            'city', 'secondary_events_count', 'avg_secondary_price_market',
            'avg_volatility_market', 'avg_listings_market',
            'high_demand_rate_market', 'unique_artists_secondary'
        ]
        
        print(f"  ✓ Markets with secondary data: {len(market_secondary)}")
        
        return market_secondary
    
    def run_final_integration(self):
        """Run complete final integration"""
        print("="*80)
        print("FINAL DATA INTEGRATION")
        print("="*80)
        
        # Load data
        self.load_existing_data()
        self.load_new_data()
        
        # Integrate DMA strength into market panel
        enhanced_market_panel = self.integrate_dma_strength_to_market_panel()
        
        # Create secondary market datasets
        secondary_events = self.create_secondary_market_dataset()
        artist_secondary = self.aggregate_secondary_market_by_artist(secondary_events)
        market_secondary = self.create_market_secondary_aggregations(secondary_events)
        
        # Enhance comprehensive artists
        enhanced_artists = self.enhance_comprehensive_artists(artist_secondary)
        
        # Save all outputs
        print("\n" + "="*80)
        print("SAVING FINAL DATASETS")
        print("="*80)
        
        outputs = {
            # Primary enhanced datasets
            'FINAL_market_panel.csv': enhanced_market_panel,
            'FINAL_comprehensive_artists.csv': enhanced_artists,
            
            # Secondary market datasets
            'secondary_market_events.csv': secondary_events,
            'artist_secondary_market.csv': artist_secondary,
            'market_secondary_aggregations.csv': market_secondary,
            
            # DMA strength (standalone)
            'dma_market_strength.csv': self.dma_strength,
        }
        
        for filename, df in outputs.items():
            filepath = f'{self.output_dir}/{filename}'
            df.to_csv(filepath, index=False)
            print(f"  ✓ {filename}: {len(df)} records, {len(df.columns)} columns")
        
        # Generate summary statistics
        self.generate_final_summary(enhanced_market_panel, enhanced_artists, secondary_events)
        
        print("\n" + "="*80)
        print("FINAL INTEGRATION COMPLETE!")
        print("="*80)
        
        return {
            'market_panel': enhanced_market_panel,
            'artists': enhanced_artists,
            'secondary_events': secondary_events,
            'artist_secondary': artist_secondary,
            'market_secondary': market_secondary
        }
    
    def generate_final_summary(self, market_panel, artists, events):
        """Generate final summary statistics"""
        print("\n" + "="*80)
        print("FINAL DATA SUMMARY")
        print("="*80)
        
        print("\n📊 MARKET PANEL (Enhanced):")
        print(f"  • Records: {len(market_panel)}")
        print(f"  • Variables: {len(market_panel.columns)}")
        print(f"  • Markets with DMA strength: {market_panel['dma_static_strength'].notna().sum()}")
        print(f"  • Avg DMA strength: {market_panel['dma_static_strength'].mean():.2f}")
        print(f"  • DMA strength range: {market_panel['dma_static_strength'].min():.0f} - {market_panel['dma_static_strength'].max():.0f}")
        
        print("\n🎤 ARTIST DATA (Enhanced):")
        print(f"  • Total artists: {len(artists)}")
        print(f"  • Variables: {len(artists.columns)}")
        print(f"  • With Last.fm data: {artists['lastfm_listeners'].notna().sum()} ({artists['lastfm_listeners'].notna().sum()/len(artists)*100:.1f}%)")
        print(f"  • With secondary market data: {artists['secondary_market_events'].notna().sum()} ({artists['secondary_market_events'].notna().sum()/len(artists)*100:.1f}%)")
        print(f"  • Avg secondary price: ${artists['avg_secondary_price'].mean():.2f}")
        
        print("\n🎫 SECONDARY MARKET EVENTS:")
        print(f"  • Total events: {len(events)}")
        print(f"  • Variables: {len(events.columns)}")
        print(f"  • Date range: {events['event_date'].min()} to {events['event_date'].max()}")
        print(f"  • Median secondary price: ${events['median_secondary_price'].median():.2f}")
        print(f"  • Avg price volatility: {events['price_volatility'].mean():.2f}")
        print(f"  • High demand events: {events['high_demand'].sum()} ({events['high_demand'].mean()*100:.1f}%)")
        
        print("\n✨ NEW VARIABLES ADDED:")
        print("  Market Panel:")
        print("    • dma_static_strength - Market strength score")
        print("    • annual_sales_avg - Historical sales average")
        print("    • avg_ticket_price_dma - DMA average price")
        print("    • strength_vs_avg - Relative market strength")
        print("    • price_premium_vs_dma - Price positioning")
        print("    • sales_momentum - Current vs historical")
        
        print("\n  Artist Data:")
        print("    • secondary_market_events - Event count")
        print("    • avg_secondary_price - Secondary market pricing")
        print("    • avg_price_volatility - Price stability")
        print("    • high_demand_rate - Demand indicator")
        print("    • premium_pricing_rate - Premium event rate")


if __name__ == "__main__":
    integrator = FinalDataIntegration()
    results = integrator.run_final_integration()
    
    print("\n\n🎉 ALL DATA INTEGRATED!")
    print("Use FINAL_market_panel.csv and FINAL_comprehensive_artists.csv for modeling")
