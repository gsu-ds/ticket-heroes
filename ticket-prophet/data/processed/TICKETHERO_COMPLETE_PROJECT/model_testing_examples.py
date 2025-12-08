"""
Model Testing Starter Kit
Quick examples for common modeling approaches with the cleaned panel data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class ModelTester:
    """Helper class for testing various models on panel data"""
    
    def __init__(self, data_path='/mnt/user-data/outputs/'):
        self.data_path = data_path
        self.results = {}
        
    def load_data(self):
        """Load the cleaned panel datasets"""
        print("Loading data...")
        self.market_panel = pd.read_csv(f'{self.data_path}market_panel.csv')
        self.artist_panel = pd.read_csv(f'{self.data_path}artist_panel.csv')
        print(f"  ✓ Market panel: {self.market_panel.shape}")
        print(f"  ✓ Artist panel: {self.artist_panel.shape}")
        
    def example_1_market_revenue_prediction(self):
        """
        Example 1: Predict market gross revenue using lagged variables
        Model type: Linear regression with panel structure
        """
        print("\n" + "="*70)
        print("EXAMPLE 1: Market Revenue Prediction")
        print("="*70)
        
        # Prepare data - drop missing lagged variables
        df = self.market_panel.dropna(subset=[
            'gross_lag1', 'tickets_lag1', 'avg_price_lag1', 'shows_lag1'
        ])
        
        print(f"\nSample size after dropping NAs: {len(df)}")
        
        # Features and target
        features = ['gross_lag1', 'tickets_lag1', 'avg_price_lag1', 
                   'shows_lag1', 'dma_rank', 'rank']
        X = df[features]
        y = df['gross']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Test multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: ${rmse:,.0f}")
            print(f"  MAE:  ${mae:,.0f}")
            print(f"  R²:   {r2:.4f}")
        
        self.results['market_revenue'] = results
        return results
    
    def example_2_market_growth_prediction(self):
        """
        Example 2: Predict market growth rates
        Model type: Growth rate prediction
        """
        print("\n" + "="*70)
        print("EXAMPLE 2: Market Growth Rate Prediction")
        print("="*70)
        
        # Prepare data
        df = self.market_panel.dropna(subset=['gross_growth', 'ticket_growth'])
        
        print(f"\nSample size: {len(df)}")
        
        # Features and target
        features = ['gross_lag1', 'tickets_lag1', 'avg_price_lag1', 
                   'shows_lag1', 'avg_price_change', 'dma_rank']
        X = df[features]
        y = df['gross_growth']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Test models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")
    
    def example_3_artist_performance(self):
        """
        Example 3: Predict artist performance metrics
        Model type: Cross-sectional regression
        """
        print("\n" + "="*70)
        print("EXAMPLE 3: Artist Performance Prediction")
        print("="*70)
        
        # Focus on artists with complete Spotify data
        df = self.artist_panel.dropna(subset=[
            'artist_popularity', 'artist_followers', 'avg_ticket_price'
        ])
        
        print(f"\nSample size with complete data: {len(df)}")
        
        if len(df) < 20:
            print("Insufficient data for modeling (need Spotify matches)")
            return
        
        # Predict number of events
        features = ['artist_popularity', 'artist_followers', 
                   'avg_track_popularity', 'total_tracks']
        X = df[features]
        y = df['num_events']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\nPredicting: Number of Events")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance.to_string(index=False))
    
    def example_4_market_fixed_effects(self):
        """
        Example 4: Panel regression with market fixed effects
        Model type: Fixed effects regression
        """
        print("\n" + "="*70)
        print("EXAMPLE 4: Fixed Effects Regression")
        print("="*70)
        
        df = self.market_panel.dropna(subset=['gross_lag1'])
        
        # Create market dummies
        market_dummies = pd.get_dummies(df['market'], prefix='market', drop_first=True)
        
        # Features including fixed effects
        base_features = ['gross_lag1', 'tickets_lag1', 'avg_price_lag1', 'shows_lag1']
        X = pd.concat([df[base_features], market_dummies], axis=1)
        y = df['gross']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Model with regularization (many features)
        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"\nWith Market Fixed Effects:")
        print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
        print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
        
        # Compare to model without fixed effects
        X_no_fe = df[base_features]
        X_train_no_fe = X_train[base_features]
        X_test_no_fe = X_test[base_features]
        
        model_no_fe = Ridge(alpha=10.0)
        model_no_fe.fit(X_train_no_fe, y_train)
        y_pred_no_fe = model_no_fe.predict(X_test_no_fe)
        
        print(f"\nWithout Market Fixed Effects:")
        print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_no_fe)):,.0f}")
        print(f"  R²:   {r2_score(y_test, y_pred_no_fe):.4f}")
    
    def example_5_time_series_market(self):
        """
        Example 5: Time series analysis for specific markets
        Model type: Time series regression
        """
        print("\n" + "="*70)
        print("EXAMPLE 5: Time Series Analysis by Market")
        print("="*70)
        
        # Focus on markets with all 4 years
        market_counts = self.market_panel.groupby('market').size()
        complete_markets = market_counts[market_counts == 4].index
        
        print(f"\nMarkets with complete time series: {len(complete_markets)}")
        
        # Select top 5 markets by average gross
        top_markets = (self.market_panel
                      .groupby('market')['gross']
                      .mean()
                      .sort_values(ascending=False)
                      .head(5)
                      .index)
        
        print(f"\nTop 5 markets:")
        for i, market in enumerate(top_markets, 1):
            market_data = self.market_panel[
                self.market_panel['market'] == market
            ].sort_values('year')
            
            avg_gross = market_data['gross'].mean()
            trend = market_data['gross'].iloc[-1] - market_data['gross'].iloc[0]
            
            print(f"  {i}. {market}: Avg ${avg_gross:,.0f}, "
                  f"Trend: ${trend:+,.0f}")
    
    def run_all_examples(self):
        """Run all example analyses"""
        self.load_data()
        
        try:
            self.example_1_market_revenue_prediction()
        except Exception as e:
            print(f"\nExample 1 failed: {e}")
        
        try:
            self.example_2_market_growth_prediction()
        except Exception as e:
            print(f"\nExample 2 failed: {e}")
        
        try:
            self.example_3_artist_performance()
        except Exception as e:
            print(f"\nExample 3 failed: {e}")
        
        try:
            self.example_4_market_fixed_effects()
        except Exception as e:
            print(f"\nExample 4 failed: {e}")
        
        try:
            self.example_5_time_series_market()
        except Exception as e:
            print(f"\nExample 5 failed: {e}")
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETE!")
        print("="*70)


if __name__ == "__main__":
    tester = ModelTester()
    tester.run_all_examples()
