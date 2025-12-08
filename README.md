# Fundamentals of Data Science Project (Fall 2025)

## Abstract 

Concert ticket prices fluctuate in an unexpected manner before the actual event. Instead of overpaying for bad tickets for the rest of our lives, we aim to build a data-driven recommendation system that predicts the optimal day to buy a ticket for a specific concert or artist, using time series forecasting and machine learning regression within a serverless (AWS Lambda) pipeline.

> A serverless machine learning pipeline that transforms real-time concert ticket data into actionable purchase recommendations using time series forecasting and predictive analytics.


## Project Overview

Concert ticket prices fluctuate in an unexpected manner before the actual event. Instead of overpaying for bad tickets for the rest of our lives, we aim to build a data-driven recommendation system that predicts the optimal day to buy a ticket for a specific concert or artist.

This project focuses on **predicting optimal purchase timing** for concert tickets to help consumers make informed decisions and avoid overpaying.

### Goals

- Predict **optimal purchase timing** for concert tickets across multiple artists and venues
- Deliver actionable insights through an **interactive recommendation system**
- Support data-driven decision making for concert-goers
- Benchmark multiple forecasting approaches using time series analysis and machine learning regression

### Methodology

Leveraging real-time concert ticket pricing data, we construct an automated serverless pipeline (AWS Lambda) with:

- **Temporal features**: Time-to-event, day-of-week, seasonal patterns
- **Price dynamics**: Historical trends, volatility measures, rolling statistics
- **Event characteristics**: Artist popularity, venue capacity, location data
- **External factors**: Demand indicators and market conditions

Models are evaluated using **R^2**, **RMSE**, **MAE**, and **directional accuracy** to assess predictive performance and real-world utility for purchase recommendations.

##  Panel Structures

### Market Panel (Primary Dataset)
- **Unit of Analysis:** Geographic market × Year
- **Time Period:** 2022-2025 (4 years))
- **Key Variables:**
  - Outcomes: `gross`, `tickets`, `shows`, `avg_price`
  - Lagged: `gross_lag1`, `tickets_lag1`, etc.
  - Growth: `gross_growth`, `ticket_growth`
  - External: Ticketmaster event counts and prices

### Artist Panel (Secondary Dataset)
- **Unit of Analysis:** Artist × Year
- **Time Period:** 2025 only
- **Key Variables:**
  - Performance: `num_events`, `avg_ticket_price`, `unique_venues`
  - Spotify: `artist_popularity`, `artist_followers`
  - Derived: `events_per_venue`


## Team

**Data Science Team** (Alphabetically): Harini Mohan, Joshua Piña

**Institution**: Georgia State University

## Contact & Support

### GitHub Issues
[Report bugs or request features](https://github.com/gsu-ds/ticket-heroes/issues)

### Email
- Joshua Piña: jpina4@student.gsu.edu
- Harini Mohan: hmohan1@student.gsu.edu

## Project Website

👉 [We Could be Prophets...](https://ticket-heroes-dokfwynss7bwctcs993ymx.streamlit.app/)
