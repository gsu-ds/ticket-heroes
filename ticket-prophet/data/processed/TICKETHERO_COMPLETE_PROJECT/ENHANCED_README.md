# COMPLETE DATA PANEL - Enhanced Edition

## 🎯 Overview

This project has been **ENHANCED** with additional data sources! You now have a comprehensive concert industry dataset that merges:

### Original Sources
- ✅ Pollstar Top 100 Markets (2022-2025)
- ✅ Spotify Artist/Track Data
- ✅ Ticketmaster Events (2025)

### NEW Enhanced Sources
- ✨ **GitHub 2016** - Historical concert data with pricing and demand
- ✨ **Last.fm** - Artist popularity and streaming metrics (12K+ artists)
- ✨ **PredictHQ** - Event prediction data (90K+ events)
- ✨ **Setlist.fm** - Concert setlist data

---

## 📊 What's NEW in Enhanced Edition

### 1. **comprehensive_artist_data.csv** (2,549 artists)
**The crown jewel** - All artist data merged in one place!

| Data Source | Match Rate | Key Variables |
|-------------|------------|---------------|
| Spotify | 100% (base) | Popularity, followers, track stats |
| Last.fm | 98% (2,496) | Listeners, playcount, tags |
| GitHub 2016 | 6% (147) | Historical events, sellout rates |
| Setlist.fm | 0% | (no US matches in sample) |

**New Variables:**
- `lastfm_listeners` - Last.fm listener count
- `lastfm_playcount` - Total plays on Last.fm
- `plays_per_listener` - Engagement metric
- `lastfm_tag_1` - Primary genre tag
- `github_events_2016` - Historical event count
- `github_avg_price` - Historical average ticket price
- `github_sellout_rate` - Historical sellout percentage
- `num_years_active` - Artist career length

### 2. **events_github_2016.csv** (3,126 events)
Historical concert data from 2016-2019

**Key Features:**
- Actual ticket sales (not just prices!)
- Sold out indicators
- Secondary market data
- Days to show (advance purchase window)

**New Variables:**
- `total_tickets` - Total tickets available
- `is_sold_out` - Binary sold out flag
- `price_min`, `price_max`, `price_avg` - Price ranges
- `tickets_per_posting` - Supply metric

### 3. **phq_clean.csv** (90,702 events)
PredictHQ event prediction data for 2025

**Categories:**
- Concerts (3,645 from this data)
- Festivals
- Sports events

**Key Variables:**
- `phq_attendance` - Predicted attendance
- `rank` - Event importance ranking
- `local_rank` - Local importance
- `duration` - Event duration in hours

### 4. **market_github_2016.csv** (20 markets)
Historical market-level data for comparison with current Pollstar

**Use Case:** Compare 2016 vs 2022-2025 market evolution

### 5. **market_phq.csv** (53,644 market observations)
Comprehensive market-level predictions from PHQ

---

## 🔥 Key Enhancements

### Artist Data Quality Improvement
**Before:** 7% of artists had Spotify data  
**After:** 98% of artists now have Last.fm streaming data!

### Historical Context
- **2016-2019 data** allows temporal comparisons
- Track how markets/artists evolved over 6-9 years
- Historical pricing for inflation adjustments

### Comprehensive Coverage
- **Original:** 1,287 artists (TM 2025 only)
- **Enhanced:** 12,754 unique artists across all sources
- **Enriched:** 2,549 artists with multi-source data

---

## 📈 NEW Analysis Possibilities

### 1. Artist Success Prediction Models
```python
# Predict concert success using streaming metrics
features = [
    'artist_popularity',       # Spotify
    'lastfm_listeners',        # Last.fm
    'lastfm_playcount',        # Last.fm
    'plays_per_listener',      # Engagement
    'github_events_2016'       # Historical performance
]
target = 'num_events'  # or ticket sales, pricing, etc.
```

### 2. Historical Evolution Analysis
```python
# Compare 2016 vs 2025
github_2016 = pd.read_csv('market_github_2016.csv')
pollstar_2025 = market_panel[market_panel['year'] == 2025]

# Analyze market growth over 9 years
```

### 3. Streaming → Concert Performance
```python
# Does streaming popularity predict concert demand?
df = comprehensive_artist_data.merge(artist_panel, on='artist_name')

correlation = df[[
    'lastfm_listeners',
    'lastfm_playcount', 
    'num_events',
    'avg_ticket_price'
]].corr()
```

### 4. Sellout Prediction
```python
# Historical sellout patterns
github = pd.read_csv('events_github_2016.csv')

# Features: price, artist popularity, advance days
# Target: is_sold_out
```

---

## 📋 Complete File List (25 files!)

### Primary Analysis Datasets
1. **market_panel.csv** - Market-year panel (400 obs)
2. **artist_panel.csv** - Artist-year panel (1,287 obs)
3. **comprehensive_artist_data.csv** - 🆕 Multi-source artist data (2,549 obs)

### Event-Level Data
4. **ticketmaster_clean.csv** - TM 2025 events (3,645)
5. **events_github_2016.csv** - 🆕 Historical events (3,126)
6. **events_setlistfm.csv** - 🆕 Setlist data (0 US events)

### Source Data (Cleaned)
7. **pollstar_clean.csv** - Pollstar markets
8. **spotify_clean.csv** - Spotify tracks
9. **github_clean.csv** - 🆕 GitHub 2016 raw
10. **lastfm_clean.csv** - 🆕 Last.fm data (12,845 artists!)
11. **phq_clean.csv** - 🆕 PredictHQ events (90,702!)
12. **setlistfm_clean.csv** - 🆕 Setlist.fm

### Aggregated Data
13. **artist_stats.csv** - Spotify aggregations
14. **artist_events.csv** - TM artist aggregations
15. **market_events.csv** - TM market aggregations
16. **market_tm_2025.csv** - 🆕 TM market summary
17. **market_github_2016.csv** - 🆕 Historical markets
18. **market_phq.csv** - 🆕 PHQ market predictions

### Code & Documentation
19. **data_cleaning_pipeline.py** - Original pipeline
20. **enhanced_data_pipeline.py** - 🆕 Enhanced pipeline
21. **model_testing_examples.py** - Example models
22. **README.md** - Project guide
23. **DATA_DICTIONARY.md** - Variable definitions
24. **EXECUTIVE_SUMMARY.md** - Quick overview
25. **FILE_INDEX.md** - File navigation

---

## 🎓 Enhanced Use Cases

### Use Case 1: Streaming-to-Concert Pipeline
**Question:** Do streaming metrics predict concert demand?

**Data:**
- `comprehensive_artist_data.csv` (Spotify + Last.fm metrics)
- `artist_panel.csv` (concert performance)

**Analysis:**
```python
df = pd.read_csv('comprehensive_artist_data.csv')
# Regress num_events on streaming metrics
```

### Use Case 2: Market Evolution (2016 → 2025)
**Question:** How have concert markets changed?

**Data:**
- `market_github_2016.csv` (historical)
- `market_panel.csv` (current)

**Analysis:**
```python
# Compare pricing, volume, sellout rates
# 9-year evolution analysis
```

### Use Case 3: Sellout Prediction
**Question:** What predicts concert sellouts?

**Data:**
- `events_github_2016.csv` (has `is_sold_out` labels!)

**Analysis:**
```python
# Use historical data to train sellout classifier
# Features: price, popularity, advance days
```

### Use Case 4: Genre-Based Analysis
**Question:** How do different genres perform?

**Data:**
- `comprehensive_artist_data.csv` (has `lastfm_tag_1` genre tags)
- `artist_panel.csv` (performance metrics)

**Analysis:**
```python
# Compare genres on pricing, attendance, geography
```

### Use Case 5: Predictive Event Intelligence
**Question:** Can we predict future market activity?

**Data:**
- `phq_clean.csv` (90K+ predicted events for 2025)
- `market_panel.csv` (historical actuals)

**Analysis:**
```python
# Validate PHQ predictions against actuals
# Build predictive models
```

---

## 📊 Key Statistics - Enhanced Edition

### Artist Coverage
- **Total unique artists across sources:** 12,754
- **Artists with comprehensive data:** 2,549
- **Spotify match rate:** 100% (base dataset)
- **Last.fm match rate:** 98% (2,496/2,549)
- **GitHub historical match rate:** 6% (147/2,549)

### Event Coverage
- **Total events across all sources:** ~100K+
  - Ticketmaster 2025: 3,645
  - GitHub 2016-2019: 3,126
  - PredictHQ 2025: 90,702
  - Setlist.fm: 4,000 (0 US in sample)

### Market Coverage
- **Pollstar markets (2022-2025):** 161 unique
- **GitHub 2016 markets:** 20
- **PHQ markets:** 53,644 market observations

### Temporal Coverage
- **Historical:** 2016-2019 (GitHub)
- **Current:** 2022-2025 (Pollstar, Ticketmaster)
- **Predictive:** 2025 (PredictHQ)

---

## 🚀 Quick Start (Enhanced)

```python
import pandas as pd

# Load the comprehensive artist dataset
artists = pd.read_csv('comprehensive_artist_data.csv')

# Check data richness
print(f"Artists with Spotify data: {artists['artist_popularity'].notna().sum()}")
print(f"Artists with Last.fm data: {artists['lastfm_listeners'].notna().sum()}")
print(f"Artists with historical data: {artists['github_events_2016'].notna().sum()}")

# Analyze streaming → concert relationship
artists_full = artists.dropna(subset=['lastfm_listeners', 'num_events'])
correlation = artists_full[[
    'lastfm_listeners', 
    'artist_followers',
    'github_events_2016'
]].corr()
```

---

## 💎 Data Quality Summary

### Excellent Quality
✅ Last.fm data: 98% coverage, 12K+ artists  
✅ Spotify data: 100% coverage (base dataset)  
✅ Pollstar markets: Complete 4-year panel  

### Good Quality
✅ GitHub 2016: 3K+ events with ticket data  
✅ PHQ predictions: 90K+ events  
✅ Ticketmaster 2025: 3.6K events  

### Limited Coverage
⚠️ GitHub historical matches: Only 6% of current artists  
⚠️ Setlist.fm: No US events in sample  
⚠️ PHQ integration: Needs further market matching  

---

## 🎯 Recommended Next Steps

1. **Explore comprehensive_artist_data.csv** first
   - 98% have streaming data!
   - Best for artist-level analysis

2. **Compare historical vs current**
   - Use events_github_2016.csv vs ticketmaster_clean.csv
   - 9-year evolution analysis

3. **Build streaming prediction models**
   - Does Last.fm predict concert success?
   - Feature engineering with engagement metrics

4. **Validate PHQ predictions**
   - Compare PHQ forecasts with actual Pollstar data
   - Build improved prediction models

5. **Geographic analysis**
   - Market-level trends across sources
   - Regional differences in pricing/demand

---

## 📞 Support

**Documentation Files:**
- Start with: `EXECUTIVE_SUMMARY.md`
- Variable definitions: `DATA_DICTIONARY.md`
- All files: `FILE_INDEX.md`

**Code Examples:**
- Basic models: `model_testing_examples.py`
- Enhanced pipeline: `enhanced_data_pipeline.py`

---

**Project Status**: ✅ Enhanced & Complete  
**Last Updated**: December 8, 2025  
**Total Data Sources**: 7 (Pollstar, Spotify, TM, GitHub, Last.fm, PHQ, Setlist.fm)  
**Total Files**: 25  
**Total Events**: ~100,000+  
**Total Artists**: 12,754 unique (2,549 enriched)
