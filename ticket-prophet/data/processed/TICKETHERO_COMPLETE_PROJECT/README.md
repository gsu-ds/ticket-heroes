# Concert Data Panel - Cleaned & Ready for Model Testing

## 📊 Project Overview

This project combines multiple data sources (Pollstar, Spotify, Ticketmaster) to create clean panel datasets for modeling and analysis of the live music industry.

## 🎯 What's Included

### Cleaned Datasets (8 files)
1. **market_panel.csv** - Main market-level panel (400 obs, 2022-2025)
2. **artist_panel.csv** - Artist-level panel (1,287 obs, 2025)
3. **pollstar_clean.csv** - Cleaned Pollstar data
4. **spotify_clean.csv** - Cleaned Spotify track data
5. **artist_stats.csv** - Aggregated artist statistics
6. **ticketmaster_clean.csv** - Cleaned Ticketmaster events
7. **market_events.csv** - Market-level event aggregations
8. **artist_events.csv** - Artist-level event aggregations

### Scripts & Documentation
- **data_cleaning_pipeline.py** - Complete cleaning pipeline (reusable)
- **model_testing_examples.py** - Example modeling workflows
- **DATA_DICTIONARY.md** - Comprehensive variable documentation

## 🚀 Quick Start

### Load the Data
```python
import pandas as pd

# Load main panels
market_panel = pd.read_csv('market_panel.csv')
artist_panel = pd.read_csv('artist_panel.csv')

# Check structure
print(market_panel.head())
print(f"Shape: {market_panel.shape}")
```

### Run Example Models
```python
# Run all example analyses
python model_testing_examples.py
```

## 📈 Panel Structures

### Market Panel (Primary Dataset)
- **Unit of Analysis:** Geographic market × Year
- **Time Period:** 2022-2025 (4 years)
- **Observations:** 400 (100 markets × 4 years)
- **Key Variables:**
  - Outcomes: `gross`, `tickets`, `shows`, `avg_price`
  - Lagged: `gross_lag1`, `tickets_lag1`, etc.
  - Growth: `gross_growth`, `ticket_growth`
  - External: Ticketmaster event counts and prices

### Artist Panel (Secondary Dataset)
- **Unit of Analysis:** Artist × Year
- **Time Period:** 2025 only
- **Observations:** 1,287 unique artists
- **Key Variables:**
  - Performance: `num_events`, `avg_ticket_price`, `unique_venues`
  - Spotify: `artist_popularity`, `artist_followers`
  - Derived: `events_per_venue`

## 🎓 Example Use Cases

### 1. Market Revenue Prediction
```python
from sklearn.ensemble import RandomForestRegressor

# Prepare data
df = market_panel.dropna(subset=['gross_lag1', 'tickets_lag1'])
X = df[['gross_lag1', 'tickets_lag1', 'avg_price_lag1', 'shows_lag1']]
y = df['gross']

# Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
# Result: R² = 0.993 on test set!
```

### 2. Growth Rate Analysis
```python
# Analyze market growth patterns
growth = market_panel.groupby('market')['gross_growth'].mean()
top_growing = growth.sort_values(ascending=False).head(10)
```

### 3. Fixed Effects Regression
```python
# Panel regression with market fixed effects
import statsmodels.formula.api as smf

model = smf.ols(
    'gross ~ gross_lag1 + tickets_lag1 + C(market)', 
    data=market_panel
).fit()
```

### 4. Artist Performance Prediction
```python
# Predict events based on Spotify popularity
artists = artist_panel.dropna(subset=['artist_popularity'])
X = artists[['artist_popularity', 'artist_followers']]
y = artists['num_events']
```

## 📊 Model Testing Results

The pipeline includes 5 pre-built examples that demonstrate:

| Example | Task | Best Model | Performance |
|---------|------|------------|-------------|
| 1 | Revenue Prediction | Random Forest | R² = 0.993 |
| 2 | Growth Prediction | Random Forest | R² = 0.124 |
| 3 | Artist Events | Random Forest | Variable |
| 4 | Fixed Effects | Ridge | R² = 0.907 |
| 5 | Time Series | Analysis | Descriptive |

## ⚠️ Data Quality Notes

### Strengths
✅ Multi-year market data (2022-2025)  
✅ Rich feature set with lagged variables  
✅ Multiple data sources for validation  
✅ Panel structure enables causal inference  

### Limitations
⚠️ Ticketmaster data only for 2025  
⚠️ ~93% of artists lack Spotify matches  
⚠️ Single year limits artist panel analysis  
⚠️ Some price data missing (~6-80% depending on source)  

### Recommendations
1. **Market analysis:** Use complete 4-year panel with lagged variables
2. **Artist analysis:** Focus on matched subset or impute missing Spotify data
3. **Temporal models:** Plenty of data for market trends, limited for artists
4. **Causal inference:** Fixed effects and diff-in-diff feasible for markets

## 🔧 Rerunning the Pipeline

To recreate the cleaned datasets:

```bash
python data_cleaning_pipeline.py
```

The pipeline will:
1. Load raw data from uploads directory
2. Clean and standardize each source
3. Create panel structures
4. Generate summary statistics
5. Save 8 output files

## 📁 File Structure

```
outputs/
├── market_panel.csv              # Main market panel
├── artist_panel.csv              # Main artist panel
├── pollstar_clean.csv           # Cleaned source data
├── spotify_clean.csv            # Cleaned source data
├── artist_stats.csv             # Aggregated data
├── ticketmaster_clean.csv       # Cleaned source data
├── market_events.csv            # Aggregated data
├── artist_events.csv            # Aggregated data
├── data_cleaning_pipeline.py    # Reproducible pipeline
├── model_testing_examples.py    # Example analyses
├── DATA_DICTIONARY.md           # Variable documentation
└── README.md                    # This file
```

## 📚 Variable Reference

### Market Panel Key Variables
- `gross` - Total gross revenue ($)
- `tickets` - Total tickets sold
- `avg_price` - Average ticket price ($)
- `shows` - Number of shows
- `gross_lag1` - Previous year's gross
- `gross_growth` - YoY growth rate
- `tm_events` - Ticketmaster event count
- `dma_rank` - Market ranking

### Artist Panel Key Variables
- `num_events` - Number of events
- `avg_ticket_price` - Average price
- `unique_venues` - Venue count
- `artist_popularity` - Spotify score (0-100)
- `artist_followers` - Spotify followers
- `avg_track_popularity` - Track score average

See **DATA_DICTIONARY.md** for complete variable definitions.

## 🤝 Contributing

To extend this analysis:
1. Add new data sources
2. Create additional derived variables
3. Implement new modeling approaches
4. Improve data matching algorithms

## 📄 License & Data Sources

**Data Sources:**
- Pollstar: Top 100 Markets (2022-2025)
- Spotify: Kaggle dataset (artist/track metadata)
- Ticketmaster: Event data (2025)

## 🎯 Next Steps

1. **Exploratory Analysis:** Visualize trends and relationships
2. **Feature Engineering:** Create industry benchmarks, seasonality indicators
3. **Advanced Models:** Neural networks, ensemble methods, Bayesian approaches
4. **Causal Inference:** Instrumental variables, RDD, synthetic controls
5. **Forecasting:** Time series models for market predictions

---

**Last Updated:** December 8, 2025  
**Pipeline Version:** 1.0  
**Contact:** See model_testing_examples.py for code examples
