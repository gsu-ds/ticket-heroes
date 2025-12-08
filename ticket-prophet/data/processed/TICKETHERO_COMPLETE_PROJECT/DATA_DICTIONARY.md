# Data Panel Documentation

## Overview
This document describes the cleaned datasets and panel structures created for model testing.

---

## Panel Datasets

### 1. Market Panel (`market_panel.csv`)
**Structure:** Market-Year panel (400 observations, 20 variables)
**Time Period:** 2022-2025
**Panel Unit:** Geographic market (city/metro area)

#### Key Variables:

**Identifiers:**
- `year`: Year (2022-2025)
- `market`: Geographic market name
- `rank`: Annual ranking within year
- `dma_rank`: Designated Market Area rank

**Outcome Variables:**
- `gross`: Total gross revenue ($)
- `tickets`: Total tickets sold
- `shows`: Number of shows
- `avg_price`: Average ticket price ($)

**Derived Variables:**
- `revenue_per_show`: Gross revenue / shows
- `tickets_per_show`: Tickets sold / shows
- `avg_price_change`: Year-over-year price change (%)

**External Data (Ticketmaster):**
- `tm_events`: Number of events from Ticketmaster
- `tm_avg_price`: Average price from Ticketmaster
- `tm_unique_artists`: Unique artists in market

**Lagged Variables:**
- `gross_lag1`: Previous year's gross
- `tickets_lag1`: Previous year's tickets
- `avg_price_lag1`: Previous year's avg price
- `shows_lag1`: Previous year's shows

**Growth Rates:**
- `gross_growth`: YoY gross revenue growth rate
- `ticket_growth`: YoY ticket sales growth rate

#### Missing Data Patterns:
- Ticketmaster variables: ~98% missing (data only for 2025)
- Lagged variables: ~40% missing (no data for 2022, first year)
- `avg_price_change`: 100% missing for 2022 only

---

### 2. Artist Panel (`artist_panel.csv`)
**Structure:** Artist-Year panel (1,287 observations, 15 variables)
**Time Period:** 2025 only (from Ticketmaster)
**Panel Unit:** Artist

#### Key Variables:

**Identifiers:**
- `artist_name`: Artist name
- `year`: Year (2025)

**Performance Variables:**
- `num_events`: Number of events/concerts
- `avg_ticket_price`: Average ticket price across events
- `unique_venues`: Number of unique venues played
- `unique_cities`: Number of unique cities visited
- `events_per_venue`: Average events per venue

**Spotify Variables:**
- `artist_popularity`: Spotify popularity score (0-100)
- `artist_followers`: Number of Spotify followers
- `avg_track_popularity`: Average popularity of artist's tracks
- `total_tracks`: Number of tracks in dataset
- `explicit_ratio`: Proportion of explicit tracks
- `avg_track_duration`: Average track duration (minutes)

**Lagged Variables:**
- `num_events_lag1`: Previous year's events (all missing - single year data)
- `avg_ticket_price_lag1`: Previous year's prices (all missing)

#### Missing Data Patterns:
- Spotify variables: ~93% missing (1,193/1,287 artists not matched)
- Lagged variables: 100% missing (single year of data)
- `avg_ticket_price`: ~6% missing

---

## Individual Cleaned Datasets

### 3. Pollstar Clean (`pollstar_clean.csv`)
- **Records:** 400
- **Source:** Pollstar Top 100 Markets (2022-2025)
- **Structure:** Annual market rankings
- **Key Use:** Primary data source for market panel

### 4. Spotify Clean (`spotify_clean.csv`)
- **Records:** 8,778 tracks
- **Artists:** 2,549 unique
- **Source:** Kaggle Spotify dataset
- **Variables:** Track metadata, artist info, album details
- **Key Use:** Artist characteristics and popularity metrics

### 5. Artist Stats (`artist_stats.csv`)
- **Records:** 2,549 artists
- **Source:** Aggregated from Spotify data
- **Variables:** Artist-level averages and counts
- **Key Use:** Artist panel enrichment

### 6. Ticketmaster Clean (`ticketmaster_clean.csv`)
- **Records:** 3,645 events
- **Time Period:** 2025 only
- **Variables:** Event details, venues, pricing, dates
- **Geographic Coverage:** Multiple cities/markets
- **Key Use:** Event-level analysis and panel enrichment

### 7. Market Events (`market_events.csv`)
- **Records:** 299 city-year combinations
- **Source:** Aggregated from Ticketmaster
- **Variables:** Event counts, average prices, unique artists by market-year
- **Key Use:** Merging with market panel

### 8. Artist Events (`artist_events.csv`)
- **Records:** 1,287 artist-year combinations
- **Source:** Aggregated from Ticketmaster
- **Variables:** Event counts, prices, venue/city diversity by artist-year
- **Key Use:** Artist panel creation

---

## Data Quality Notes

### Strengths:
1. **Multi-year coverage** for market data (2022-2025)
2. **Rich variables** including both outcomes and potential predictors
3. **Multiple data sources** allowing validation and enrichment
4. **Panel structure** enables time-series and fixed effects models

### Limitations:
1. **Ticketmaster data** only available for 2025
2. **Artist matching** between Spotify and Ticketmaster incomplete (~93% unmatched)
3. **Market name variations** may cause merging issues
4. **Price data** has significant missingness in Ticketmaster
5. **Single year** for artist panel limits temporal analysis

### Recommendations:
1. **Market Panel Analysis:**
   - Use 2022-2025 data for temporal models
   - Handle missing Ticketmaster data (drop or impute)
   - Consider market fixed effects
   - Use lagged variables for dynamic models

2. **Artist Panel Analysis:**
   - Focus on cross-sectional analysis (2025 only)
   - Address Spotify matching issues
   - Use robust methods for missing data
   - Consider artist-genre clustering

3. **Model Testing Approaches:**
   - Panel regression with fixed effects
   - Difference-in-differences designs
   - Time series forecasting
   - Cross-sectional prediction models

---

## Variable Definitions Quick Reference

### Market Panel
| Variable | Type | Description | Missing |
|----------|------|-------------|---------|
| year | int | Year (2022-2025) | 0% |
| market | str | Market name | 0% |
| gross | int | Total revenue ($) | 0% |
| tickets | int | Total tickets | 0% |
| avg_price | float | Avg ticket price | 0% |
| shows | int | Number of shows | 0% |
| gross_growth | float | YoY growth rate | 40% |
| tm_events | float | TM event count | 98% |

### Artist Panel
| Variable | Type | Description | Missing |
|----------|------|-------------|---------|
| artist_name | str | Artist name | 0% |
| num_events | int | Event count | 0% |
| avg_ticket_price | float | Avg price | 6% |
| unique_venues | int | Venue count | 0% |
| artist_popularity | float | Spotify score | 93% |
| artist_followers | float | Spotify followers | 93% |

---

## Example Analysis Workflows

### Workflow 1: Market-Level Revenue Prediction
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
panel = pd.read_csv('market_panel.csv')

# Prepare features (drop rows with missing lagged variables)
panel_clean = panel.dropna(subset=['gross_lag1', 'tickets_lag1', 'shows_lag1'])

# Features and target
X = panel_clean[['gross_lag1', 'tickets_lag1', 'avg_price_lag1', 
                 'shows_lag1', 'dma_rank']]
y = panel_clean['gross']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

### Workflow 2: Artist Performance Analysis
```python
import pandas as pd

# Load data
artists = pd.read_csv('artist_panel.csv')

# Focus on artists with complete Spotify data
artists_complete = artists.dropna(subset=['artist_popularity'])

# Analyze relationship between popularity and performance
correlation = artists_complete[[
    'artist_popularity', 'artist_followers', 'num_events',
    'avg_ticket_price', 'unique_cities'
]].corr()
```

### Workflow 3: Time Series Forecasting
```python
import pandas as pd

# Load data
panel = pd.read_csv('market_panel.csv')

# Focus on top markets with complete time series
top_markets = ['New York', 'Los Angeles', 'Las Vegas']
ts_data = panel[panel['market'].isin(top_markets)]

# Pivot to wide format for time series
ts_wide = ts_data.pivot(index='year', columns='market', values='gross')

# Use for ARIMA, Prophet, or other time series models
```

---

## Next Steps

1. **Exploratory Data Analysis:** Visualize trends, distributions, relationships
2. **Feature Engineering:** Create interaction terms, polynomials, industry benchmarks
3. **Model Selection:** Choose appropriate models based on research questions
4. **Validation:** Use proper cross-validation for panel data
5. **Iteration:** Refine based on model diagnostics and domain knowledge
