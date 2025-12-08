"""
TicketHero Prophet - Presentation Models
Aligned with presentation structure: KNN baseline, Decision Tree, Random Forest, XGBoost, CatBoost
Goal: Predict ticket prices using market features + historical data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

class TicketHeroProphet:
    """
    Main modeling pipeline for TicketHero Prophet presentation
    Matches slide 7: Baseline (KNN) + Models (DT, RF, XGB, CatBoost)
    """
    
    def __init__(self):
        self.results = {}
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """
        Load your comprehensive datasets and prepare for modeling
        Matches slide 6: Feature Engineering + Data Collection
        """
        print("="*80)
        print("TICKETHERO PROPHET - DATA LOADING")
        print("="*80)
        
        # Load the FINAL datasets with all features
        print("\n1. Loading comprehensive datasets...")
        self.market_panel = pd.read_csv('/mnt/user-data/outputs/FINAL_market_panel.csv')
        self.artists = pd.read_csv('/mnt/user-data/outputs/FINAL_comprehensive_artists.csv')
        self.secondary = pd.read_csv('/mnt/user-data/outputs/secondary_market_events.csv')
        
        print(f"   ✓ Market panel: {self.market_panel.shape}")
        print(f"   ✓ Artist data: {self.artists.shape}")
        print(f"   ✓ Secondary market: {self.secondary.shape}")
        
        # Choose modeling approach based on available data
        print("\n2. Preparing modeling datasets...")
        self.prepare_market_price_prediction()
        self.prepare_secondary_market_prediction()
        
    def prepare_market_price_prediction(self):
        """
        Scenario 1: Market-level average ticket price prediction
        Uses: DMA strength, historical data, market characteristics
        """
        print("\n   → Market-Level Price Prediction Dataset")
        
        df = self.market_panel.copy()
        
        # Filter to complete cases with DMA strength
        df = df.dropna(subset=['dma_static_strength', 'avg_price'])
        
        # Feature selection (matching your presentation themes)
        self.market_features = [
            # Market strength (YOUR new data!)
            'dma_static_strength',
            'strength_vs_avg',
            'sales_momentum',
            
            # Historical performance
            'gross_lag1',
            'tickets_lag1',
            'shows_lag1',
            
            # Market characteristics
            'dma_rank',
            'rank',
            
            # Growth indicators
            'gross_growth',
            'ticket_growth'
        ]
        
        # Keep only available features
        self.market_features = [f for f in self.market_features if f in df.columns]
        
        df_clean = df[self.market_features + ['avg_price']].dropna()
        
        self.X_market = df_clean[self.market_features]
        self.y_market = df_clean['avg_price']
        
        print(f"      • Features: {len(self.market_features)}")
        print(f"      • Samples: {len(self.X_market)}")
        print(f"      • Target: avg_price (${self.y_market.mean():.2f} avg)")
        
    def prepare_secondary_market_prediction(self):
        """
        Scenario 2: Event-level secondary market price prediction
        Uses: Secondary market features, volatility, demand signals
        """
        print("\n   → Secondary Market Price Prediction Dataset")
        
        df = self.secondary.copy()
        
        # Filter to complete cases
        df = df.dropna(subset=['median_secondary_price'])
        
        # Feature selection
        self.secondary_features = [
            'active_listings_count',
            'price_volatility',
            'price_range',
            'avg_daily_price_change',
            'valid_price_observations',
            'observation_window_days',
            'high_demand',
            'high_volatility',
            'premium_pricing'
        ]
        
        # Keep only available features
        self.secondary_features = [f for f in self.secondary_features if f in df.columns]
        
        df_clean = df[self.secondary_features + ['median_secondary_price']].dropna()
        
        self.X_secondary = df_clean[self.secondary_features]
        self.y_secondary = df_clean['median_secondary_price']
        
        print(f"      • Features: {len(self.secondary_features)}")
        print(f"      • Samples: {len(self.X_secondary)}")
        print(f"      • Target: median_secondary_price (${self.y_secondary.mean():.2f} avg)")
        
    def train_baseline_knn(self, X, y, scenario_name):
        """
        BASELINE MODEL: K-Nearest Neighbors
        Matches slide 7: "Baseline: KNN"
        """
        print(f"\n   BASELINE: K-Nearest Neighbors ({scenario_name})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features (important for KNN!)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train KNN
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = knn.predict(X_test_scaled)
        
        # Metrics (matching slide 8: RMSE, MAE, R²)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"      RMSE: ${rmse:.2f}")
        print(f"      MAE:  ${mae:.2f}")
        print(f"      R²:   {r2:.4f}")
        
        return {
            'model': 'KNN (Baseline)',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_obj': knn
        }
    
    def train_decision_tree(self, X, y, scenario_name):
        """
        MODEL 1: Decision Tree
        Matches slide 7: "Models: Decision Tree"
        """
        print(f"\n   MODEL 1: Decision Tree ({scenario_name})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Decision Tree
        dt = DecisionTreeRegressor(max_depth=10, min_samples_split=20, random_state=42)
        dt.fit(X_train, y_train)
        
        # Predictions
        y_pred = dt.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"      RMSE: ${rmse:.2f}")
        print(f"      MAE:  ${mae:.2f}")
        print(f"      R²:   {r2:.4f}")
        
        # Feature importance (for slide 8)
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': dt.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n      Top 3 Features:")
        for idx, row in feature_imp.head(3).iterrows():
            print(f"        {row['feature']}: {row['importance']:.4f}")
        
        return {
            'model': 'Decision Tree',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_obj': dt,
            'feature_importance': feature_imp
        }
    
    def train_random_forest(self, X, y, scenario_name):
        """
        MODEL 2: Random Forest
        Matches slide 7: "Models: Random Forest"
        """
        print(f"\n   MODEL 2: Random Forest ({scenario_name})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"      RMSE: ${rmse:.2f}")
        print(f"      MAE:  ${mae:.2f}")
        print(f"      R²:   {r2:.4f}")
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n      Top 3 Features:")
        for idx, row in feature_imp.head(3).iterrows():
            print(f"        {row['feature']}: {row['importance']:.4f}")
        
        return {
            'model': 'Random Forest',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_obj': rf,
            'feature_importance': feature_imp
        }
    
    def train_xgboost(self, X, y, scenario_name):
        """
        MODEL 3: XGBoost
        Matches slide 7: "Models: XGBoost"
        """
        print(f"\n   MODEL 3: XGBoost ({scenario_name})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"      RMSE: ${rmse:.2f}")
        print(f"      MAE:  ${mae:.2f}")
        print(f"      R²:   {r2:.4f}")
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n      Top 3 Features:")
        for idx, row in feature_imp.head(3).iterrows():
            print(f"        {row['feature']}: {row['importance']:.4f}")
        
        return {
            'model': 'XGBoost',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_obj': xgb_model,
            'feature_importance': feature_imp
        }
    
    def train_catboost(self, X, y, scenario_name):
        """
        MODEL 4: CatBoost
        Matches slide 7: "Models: CatBoost"
        """
        print(f"\n   MODEL 4: CatBoost ({scenario_name})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train CatBoost
        cb = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        cb.fit(X_train, y_train)
        
        # Predictions
        y_pred = cb.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"      RMSE: ${rmse:.2f}")
        print(f"      MAE:  ${mae:.2f}")
        print(f"      R²:   {r2:.4f}")
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': cb.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n      Top 3 Features:")
        for idx, row in feature_imp.head(3).iterrows():
            print(f"        {row['feature']}: {row['importance']:.4f}")
        
        return {
            'model': 'CatBoost',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_obj': cb,
            'feature_importance': feature_imp
        }
    
    def run_all_models(self, X, y, scenario_name):
        """
        Run all models (baseline + 4 models) for a scenario
        """
        print(f"\n{'='*80}")
        print(f"TRAINING ALL MODELS - {scenario_name}")
        print('='*80)
        
        results = []
        
        # Baseline
        results.append(self.train_baseline_knn(X, y, scenario_name))
        
        # Models
        results.append(self.train_decision_tree(X, y, scenario_name))
        results.append(self.train_random_forest(X, y, scenario_name))
        results.append(self.train_xgboost(X, y, scenario_name))
        results.append(self.train_catboost(X, y, scenario_name))
        
        return results
    
    def create_results_summary(self, results, scenario_name):
        """
        Create summary table for presentation
        """
        df_results = pd.DataFrame(results)[['model', 'rmse', 'mae', 'r2']]
        df_results = df_results.sort_values('r2', ascending=False)
        
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY - {scenario_name}")
        print('='*80)
        print(df_results.to_string(index=False))
        print()
        
        # Best model
        best = df_results.iloc[0]
        print(f"🏆 BEST MODEL: {best['model']}")
        print(f"   • R² = {best['r2']:.4f}")
        print(f"   • RMSE = ${best['rmse']:.2f}")
        print(f"   • MAE = ${best['mae']:.2f}")
        
        return df_results
    
    def run_presentation_pipeline(self):
        """
        Complete pipeline for your presentation
        Runs both scenarios and generates results
        """
        print("\n")
        print("█"*80)
        print("█" + " "*78 + "█")
        print("█" + "  TICKETHERO PROPHET - MODELING PIPELINE".center(78) + "█")
        print("█" + "  Presentation-Ready Results".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        # Load data
        self.load_and_prepare_data()
        
        # SCENARIO 1: Market-level price prediction
        print("\n\n" + "="*80)
        print("SCENARIO 1: MARKET-LEVEL PRICE PREDICTION")
        print("="*80)
        print("Question: Can we predict average ticket prices using market strength?")
        print("Uses: DMA strength, historical sales, market characteristics")
        
        market_results = self.run_all_models(
            self.X_market, 
            self.y_market, 
            "Market Prices"
        )
        market_summary = self.create_results_summary(market_results, "Market Prices")
        
        # SCENARIO 2: Secondary market price prediction
        if len(self.X_secondary) > 50:
            print("\n\n" + "="*80)
            print("SCENARIO 2: SECONDARY MARKET PRICE PREDICTION")
            print("="*80)
            print("Question: Can we predict secondary prices using demand signals?")
            print("Uses: Volatility, listings, demand indicators")
            
            secondary_results = self.run_all_models(
                self.X_secondary,
                self.y_secondary,
                "Secondary Market Prices"
            )
            secondary_summary = self.create_results_summary(
                secondary_results, 
                "Secondary Market Prices"
            )
        else:
            print("\n\n⚠️  Insufficient secondary market data for full modeling")
            secondary_summary = None
        
        # Save results
        market_summary.to_csv('/mnt/user-data/outputs/PRESENTATION_market_results.csv', index=False)
        print(f"\n✓ Saved: PRESENTATION_market_results.csv")
        
        if secondary_summary is not None:
            secondary_summary.to_csv('/mnt/user-data/outputs/PRESENTATION_secondary_results.csv', index=False)
            print(f"✓ Saved: PRESENTATION_secondary_results.csv")
        
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + "  MODELING COMPLETE - READY FOR PRESENTATION!".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        return market_summary, secondary_summary


if __name__ == "__main__":
    # Run the complete pipeline
    prophet = TicketHeroProphet()
    market_results, secondary_results = prophet.run_presentation_pipeline()
    
    print("\n\n📊 USE THESE RESULTS IN YOUR PRESENTATION!")
    print("   • Market prediction results: PRESENTATION_market_results.csv")
    if secondary_results is not None:
        print("   • Secondary market results: PRESENTATION_secondary_results.csv")
