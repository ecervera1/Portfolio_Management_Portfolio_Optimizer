import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def calculate_sector_statistics(returns):
    """
    Calculate sector-level statistics such as mean returns, standard deviations, and correlations.
    """
    sector_mean_returns = returns.mean()
    sector_covariance_matrix = returns.cov()
    return sector_mean_returns, sector_covariance_matrix

def sector_portfolio_optimization(mean_returns, covariance_matrix, sector_constraints):
    """
    Perform portfolio optimization for each sector.
    """
    num_assets = len(mean_returns)
    bounds = [(0, 1) for _ in range(num_assets)]
    
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    constraints += sector_constraints

    initial_guess = [1. / num_assets for _ in range(num_assets)]
    optimal_weights = minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimal_weights.x

def sector_level_portfolio_optimization(sector_returns, sector_constraints):
    """
    Perform sector-level portfolio optimization.
    """
    sector_weights = {}
    
    for sector, returns in sector_returns.items():
        mean_returns, covariance_matrix = calculate_sector_statistics(returns)
        optimal_weights = sector_portfolio_optimization(mean_returns, covariance_matrix, sector_constraints)
        sector_weights[sector] = optimal_weights
        
    return sector_weights

# Example usage
# Assume you have historical returns data for assets within each sector stored in separate dataframes
# Each dataframe should have columns representing asset returns and rows representing time periods

# Define Streamlit app

st.title("Sector-Level Portfolio Optimizer")

# Sector returns data (example)
sector_returns = {
    'Technology': pd.DataFrame(np.random.randn(100, 5), columns=['AAPL', 'MSFT', 'GOOGL', 'FB', 'AMZN']),
    'Finance': pd.DataFrame(np.random.randn(100, 5), columns=['JPM', 'BAC', 'WFC', 'C', 'GS']),
    # Add more sectors as needed
}

# Define sector-level constraints
# Example: Sector exposure limits (weights sum up to 1)
sector_constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights equals 1
    # Add more constraints as needed
]

# Perform sector-level portfolio optimization
sector_weights = sector_level_portfolio_optimization(sector_returns, sector_constraints)

# Display optimized sector weights
for sector, weights in sector_weights.items():
    st.write(f"Sector: {sector}")
    st.write("Optimal Weights:", weights)

