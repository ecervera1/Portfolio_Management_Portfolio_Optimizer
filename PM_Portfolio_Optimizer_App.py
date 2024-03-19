import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Function to fetch historical data
def fetch_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    valid_tickers = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[ticker] = df['Adj Close']
                valid_tickers.append(ticker)
            else:
                st.warning(f"No data available for {ticker}. Skipping...")
        except Exception as e:
            st.error(f"Failed to fetch data for {ticker}: {e}")
    return data, valid_tickers

# Portfolio Analysis Functions
def calculate_sharpe_ratio(returns, volatility):
    risk_free_rate = 0.02  # Assuming a risk-free rate of 2%
    return (returns - risk_free_rate) / volatility

def get_portfolio_statistics(weights, daily_returns, cov_matrix):
    portfolio_return = np.dot(weights, daily_returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility)
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe(weights, daily_returns, cov_matrix):
    return -get_portfolio_statistics(weights, daily_returns, cov_matrix)[2]

def optimize_portfolio(daily_returns, cov_matrix, valid_tickers):
    bounds = [(0, 1) for _ in range(len(valid_tickers))]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_guess = [1. / len(valid_tickers) for _ in range(len(valid_tickers))]
    result = minimize(negative_sharpe, initial_guess, args=(daily_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def run_analysis(tickers, start_date, end_date):
    data, valid_tickers = fetch_data(tickers, start_date, end_date)
    if data.empty:
        st.error("No valid data found for any ticker symbols.")
        return

    daily_returns = data.pct_change().dropna()
    cov_matrix = daily_returns.cov()
    optimal_weights = optimize_portfolio(daily_returns, cov_matrix, valid_tickers)
    
    portfolio_return, portfolio_volatility, sharpe_ratio = get_portfolio_statistics(optimal_weights, daily_returns, cov_matrix)

    ticker_weights = dict(zip(valid_tickers, optimal_weights))
    st.write("### Suggested Ticker Weights")
    st.table(pd.DataFrame.from_dict(ticker_weights, orient='index', columns=['Weight']))

    st.write(f'**Annual Return:** {portfolio_return:.2f}')
    st.write(f'**Daily Return:** {np.dot(optimal_weights, daily_returns.mean()):.4f}')
    st.write(f'**Risk (Standard Deviation):** {portfolio_volatility:.2f}')
    st.write(f'**Sharpe Ratio:** {sharpe_ratio:.2f}')

    # Plot the efficient frontier
    plot_efficient_frontier(daily_returns, cov_matrix, portfolio_return, portfolio_volatility, valid_tickers)

def plot_efficient_frontier(daily_returns, cov_matrix, optimal_return, optimal_volatility, valid_tickers):
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(valid_tickers))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility, _ = get_portfolio_statistics(weights, daily_returns, cov_matrix)
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = calculate_sharpe_ratio(portfolio_return, portfolio_volatility)

    plt.figure(figsize=(10, 8))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=100)
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Returns')
    st.pyplot(plt)
if st.sidebar.checkbox('Portflio Optimizer 1', value=False):
    # Initialize the Streamlit app
    st.title('Advanced Portfolio Optimizer')
    st.markdown('''
    This tool allows you to analyze and optimize your investment portfolio. Enter your tickers and select the date range to get started.
    ''')
    
    # User inputs for the tickers
    default_tickers = ['AAPL', 'GOOGL', 'MSFT']
    tickers = st.text_input('Enter the tickers separated by commas', value=','.join(default_tickers)).upper().split(',')
    
    # User input for the date range
    start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('2021-01-01'))
    
    # Check if the end date is after the start date
    if start_date >= end_date:
        st.error('Error: End date must be after start date.')
    else:
        # Button to run the analysis
        if st.button('Run Analysis'):
            run_analysis(tickers, start_date, end_date)

if st.sidebar.checkbox('Portflio Optimizer 2', value=False):
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import streamlit as st
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from pypfopt import expected_returns, risk_models, EfficientFrontier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Function to fetch historical data
    def fetch_data(tickers, start_date, end_date):
        data = pd.DataFrame()
        valid_tickers = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    data[ticker] = df['Adj Close']
                    valid_tickers.append(ticker)
                else:
                    st.warning(f"No data available for {ticker}. Skipping...")
            except Exception as e:
                st.error(f"Failed to fetch data for {ticker}: {e}")
        return data, valid_tickers
    
    # Portfolio Analysis Functions
    def calculate_sharpe_ratio(returns, volatility, risk_free_rate):
        return (returns - risk_free_rate) / volatility
    
    def get_portfolio_statistics(weights, daily_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, daily_returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate)
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_portfolio(daily_returns, cov_matrix, valid_tickers):
        # PyPortfolioOpt: Use Expected Returns and Covariance Matrix
        mu = expected_returns.mean_historical_return(daily_returns)
        S = risk_models.sample_cov(daily_returns)
    
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        return np.array([weights[ticker] for ticker in valid_tickers])
    
    def run_analysis(tickers, start_date, end_date, risk_free_rate):
        data, valid_tickers = fetch_data(tickers, start_date, end_date)
        if data.empty:
            st.error("No valid data found for any ticker symbols.")
            return
    
        daily_returns = data.pct_change().dropna()
        cov_matrix = daily_returns.cov()
        optimal_weights = optimize_portfolio(daily_returns, cov_matrix, valid_tickers)
        
        portfolio_return, portfolio_volatility, sharpe_ratio = get_portfolio_statistics(optimal_weights, daily_returns, cov_matrix, risk_free_rate)
    
        ticker_weights = dict(zip(valid_tickers, optimal_weights))
        st.write("### Suggested Ticker Weights")
        st.table(pd.DataFrame.from_dict(ticker_weights, orient='index', columns=['Weight']))
    
        st.write(f'**Annual Return:** {portfolio_return:.2f}')
        st.write(f'**Daily Return:** {np.dot(optimal_weights, daily_returns.mean()):.4f}')
        st.write(f'**Risk (Standard Deviation):** {portfolio_volatility:.2f}')
        st.write(f'**Sharpe Ratio:** {sharpe_ratio:.2f}')
    
        # Plot the efficient frontier
        plot_efficient_frontier(daily_returns, cov_matrix, portfolio_return, portfolio_volatility, valid_tickers)
    
    def plot_efficient_frontier(daily_returns, cov_matrix, optimal_return, optimal_volatility, valid_tickers):
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(len(valid_tickers))
            weights /= np.sum(weights)
            portfolio_return, portfolio_volatility, _ = get_portfolio_statistics(weights, daily_returns, cov_matrix, 0.02)
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = calculate_sharpe_ratio(portfolio_return, portfolio_volatility, 0.02)
    
        plt.figure(figsize=(10, 8))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=100)
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Returns')
        st.pyplot(plt)
    
    # Initialize the Streamlit app
    st.title('Advanced Portfolio Optimizer')
    st.markdown('''
    This tool allows you to analyze and optimize your investment portfolio. Enter your tickers, select the date range, and specify the risk-free rate to get started.
    ''')
    
    # User inputs for the tickers
    default_tickers = ['AAPL', 'GOOGL', 'MSFT']
    tickers = st.text_input('Enter the tickers separated by commas', value=','.join(default_tickers)).upper().split(',')
    
    # User input for the date range
    start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('2021-01-01'))
    
    # User input for the risk-free rate
    risk_free_rate = st.number_input('Risk-Free Rate (%)', min_value=0.0, max_value=100.0, value=2.0) / 100.0
    
    # Check if the end date is after the start date
    if start_date >= end_date:
        st.error('Error: End date must be after start date.')
    else:
        # Button to run the analysis
        if st.button('Run Analysis'):
            run_analysis(tickers, start_date, end_date, risk_free_rate)

if st.sidebar.checkbox('Portfolio Optimizer 3', value = False):
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import streamlit as st
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from pypfopt import expected_returns, risk_models, EfficientFrontier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Function to fetch historical data
    def fetch_data(tickers, start_date, end_date):
        data = pd.DataFrame()
        valid_tickers = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    data[ticker] = df['Adj Close']
                    valid_tickers.append(ticker)
                else:
                    st.warning(f"No data available for {ticker}. Skipping...")
            except Exception as e:
                st.error(f"Failed to fetch data for {ticker}: {e}")
        return data, valid_tickers
    
    # Portfolio Analysis Functions
    def calculate_sharpe_ratio(returns, volatility, risk_free_rate):
        return (returns - risk_free_rate) / volatility
    
    def get_portfolio_statistics(weights, daily_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, daily_returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate)
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe(weights, daily_returns, cov_matrix, risk_free_rate):
        return -get_portfolio_statistics(weights, daily_returns, cov_matrix, risk_free_rate)[2]
    
    def optimize_portfolio(daily_returns, cov_matrix, valid_tickers):
        # PyPortfolioOpt: Use Expected Returns and Covariance Matrix
        mu = expected_returns.mean_historical_return(daily_returns)
        S = risk_models.sample_cov(daily_returns)
    
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        return np.array([weights[ticker] for ticker in valid_tickers])
    
    def rebalance_portfolio(previous_weights, daily_returns, risk_free_rate):
        # Example: Rebalance quarterly
        rebalanced_weights = previous_weights
        return rebalanced_weights
    
    def monte_carlo_simulation(daily_returns, cov_matrix, num_simulations):
        simulation_results = []
        for _ in range(num_simulations):
            weights = np.random.random(len(valid_tickers))
            weights /= np.sum(weights)
            portfolio_return, portfolio_volatility, _ = get_portfolio_statistics(weights, daily_returns, cov_matrix, risk_free_rate)
            simulation_results.append((weights, portfolio_return, portfolio_volatility))
        return simulation_results
    
    def run_analysis(tickers, start_date, end_date, risk_free_rate, num_simulations):
        data, valid_tickers = fetch_data(tickers, start_date, end_date)
        if data.empty:
            st.error("No valid data found for any ticker symbols.")
            return
    
        daily_returns = data.pct_change().dropna()
        cov_matrix = daily_returns.cov()
        optimal_weights = optimize_portfolio(daily_returns, cov_matrix, valid_tickers)
        
        portfolio_return, portfolio_volatility, sharpe_ratio = get_portfolio_statistics(optimal_weights, daily_returns, cov_matrix, risk_free_rate)
    
        ticker_weights = dict(zip(valid_tickers, optimal_weights))
        st.write("### Suggested Ticker Weights")
        st.table(pd.DataFrame.from_dict(ticker_weights, orient='index', columns=['Weight']))
    
        st.write(f'**Annual Return:** {portfolio_return:.2f}')
        st.write(f'**Daily Return:** {np.dot(optimal_weights, daily_returns.mean()):.4f}')
        st.write(f'**Risk (Standard Deviation):** {portfolio_volatility:.2f}')
        st.write(f'**Sharpe Ratio:** {sharpe_ratio:.2f}')
    
        # Plot the efficient frontier
        plot_efficient_frontier(daily_returns, cov_matrix, portfolio_return, portfolio_volatility, valid_tickers)
    
        # Run Monte Carlo Simulation
        simulation_results = monte_carlo_simulation(daily_returns, cov_matrix, num_simulations)
        plot_monte_carlo_simulation(simulation_results, portfolio_return, portfolio_volatility)
    
        # Additional Functions for Plotting
        import numpy as np
        import matplotlib.pyplot as plt
        from pypfopt import plotting
        
        # Enhanced Efficient Frontier Visualization
        def plot_efficient_frontier(ef, daily_returns, cov_matrix, valid_tickers):
            mu = expected_returns.mean_historical_return(daily_returns)
            S = risk_models.sample_cov(daily_returns)
            
            fig, ax = plt.subplots()
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
        
            # Find the portfolio with maximum Sharpe ratio
            ef.max_sharpe()
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe Ratio")
        
            # Plot individual asset points
            for i, txt in enumerate(valid_tickers):
                ax.annotate(txt, (np.sqrt(S[i][i]), mu[i]), xytext=(5,5), textcoords='offset points')
        
            ax.set_title("Efficient Frontier with Individual Assets")
            ax.legend()
            plt.show()
    
        import matplotlib.pyplot as plt

        def plot_monte_carlo_simulation(simulation_results, portfolio_return, portfolio_volatility):
            # Extract returns, volatilities, and Sharpe ratios from the simulation results
            returns = [result[1] for result in simulation_results]
            volatilities = [result[2] for result in simulation_results]
        
            # Find the portfolio with the highest Sharpe ratio
            sharpe_ratios = [(result[1] - risk_free_rate) / result[2] for result in simulation_results]
            max_sharpe_idx = np.argmax(sharpe_ratios)
            max_sharpe_return = returns[max_sharpe_idx]
            max_sharpe_volatility = volatilities[max_sharpe_idx]
        
            plt.figure(figsize=(10, 6))
            plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis')
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel('Volatility (Std. Deviation)')
            plt.ylabel('Expected Returns')
            plt.title('Monte Carlo Simulation of Portfolio Optimization')
        
            # Highlight the portfolio with the highest Sharpe ratio
            plt.scatter(max_sharpe_volatility, max_sharpe_return, c='red', marker='*', s=200, label='Maximum Sharpe ratio')
            plt.legend(labelspacing=0.8)
        
            plt.show()

    
    # Initialize the Streamlit app
    st.title('Advanced Portfolio Optimizer')
    st.markdown('''
    This tool allows you to analyze and optimize your investment portfolio. Enter your tickers, select the date range, specify the risk-free rate, and set the number of Monte Carlo simulations to get started.
    ''')
    
    # User inputs for the tickers
    default_tickers = ['AAPL', 'GOOGL', 'MSFT']
    tickers = st.text_input('Enter the tickers separated by commas', value=','.join(default_tickers)).upper().split(',')
    
    # User input for the date range
    start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('2021-01-01'))
    
    # User input for the risk-free rate
    risk_free_rate = st.number_input('Risk-Free Rate (%)', min_value=0.0, max_value=100.0, value=2.0) / 100.0
    
    # User input for the number of Monte Carlo simulations
    num_simulations = st.number_input('Number of Monte Carlo Simulations', min_value=1, max_value=10000, value=1000)
    
    # Check if the end date is after the start date
    if start_date >= end_date:
        st.error('Error: End date must be after start date.')
    else:
        # Button to run the analysis
        if st.button('Run Analysis'):
            run_analysis(tickers, start_date, end_date, risk_free_rate, num_simulations)
if st.sidebar.checkbox('Portflio Optimizer 4', value=False):
    
if st.sidebar.checkbox('Portflio Optimizer 5', value=False):
    
