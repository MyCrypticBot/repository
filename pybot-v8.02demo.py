import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Function to get the current time in a specific format
def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Add these lines where you start the process
start_time = get_current_time()
print(f"Process started at {start_time}")

# Send email
def send_email(subject, body):
    sender_email = "mycrypticbot@gmail.com"  # Replace with your email address
    receiver_email = "jcall60@icloud.com"  # Replace with your email address (or a different recipient)
    password = "fklq dvhe fbnt aofn"  # Use your Gmail password or app-specific password

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Attach the email body to the email
    msg.attach(MIMEText(body, 'plain'))

    # Connect to the Gmail SMTP server and send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
            server.login(sender_email, password)  # Login to your Gmail account
            server.sendmail(sender_email, receiver_email, msg.as_string())  # Send the email
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Function to check if a date is a business day
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

# Function to add business days to a date
def add_business_days(start_date, business_days):
    current_date = start_date
    while business_days > 0:
        current_date += timedelta(days=1)
        if is_business_day(current_date):
            business_days -= 1
    return current_date


def get_last_business_day(year, option_data_dates):
    """ Find the last business day of the given year from the available dates. """
    year_end = datetime(year, 12, 31)
    year_dates = [date for date in option_data_dates if date.year == year]
    return max(date for date in year_dates if date <= year_end)

# Function to backtest the strategy with different parameters
def backtest_strategy(initial_capital, leverage, sell_threshold, stop_loss_threshold, trading_fee, margin_fee_annual_rate, print_statements=True):
    capital = initial_capital
    positions = 0
    buy_price = 0
    buy_time = None
    num_orders = 0
    total_fees_paid = 0
    final_value = initial_capital
    successful_trades = 0
    stop_losses = 0
    stop_losses_by_year = {}  # Dictionary to track stop-losses by year
    successful_trades_by_year = {}  # Dictionary to track successful trades by year
    holding_periods = []
    margin_fees_paid = []
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    settlement_dates = []  # Initialize settlement_dates list
    portfolio_values = {option_data_all_dates.index[0]: initial_capital}  # Dictionary to track portfolio value over time

    for i in range(1, len(ret)):
        date = option_data_all_dates.index[i]
        current_price = option_data_all_dates['close'].iloc[i]

        # Check if current date is a business day and if any trades can settle today
        if not is_business_day(date):
            continue
        settlement_dates = [d for d in settlement_dates if d > date]

        if positions > 0:
            holding_period = (date - buy_time).days
            margin_fee = (positions * buy_price) * (leverage - 1) / leverage * margin_fee_annual_rate * (holding_period / 365)
            
            if current_price >= (1 + sell_threshold) * buy_price or (ret.iloc[i] >= percentile_95 and not buy_signal.iloc[i]):
                sell_amount = positions * current_price
                fee_paid = sell_amount * trading_fee
                total_fees_paid += fee_paid + margin_fee
                capital += sell_amount - fee_paid - margin_fee
                percent_change = (current_price - buy_price) / buy_price * 100
                if print_statements:
                    print(f"Sold   {positions} positions at ${current_price:.2f} on {date}, total amount: ${sell_amount:.2f}, percent change: {percent_change:.2f}%, +{(current_price - buy_price)*positions}")
                positions = 0
                buy_price = 0
                num_orders += 1
                sell_dates.append(date)
                sell_prices.append(current_price)
                holding_periods.append(holding_period)
                margin_fees_paid.append(margin_fee)
                successful_trades += 1
                successful_trades_by_year[date.year] = successful_trades_by_year.get(date.year, 0) + 1  # Update successful trades count for the year
                settlement_dates.append(add_business_days(date, 2))

            elif current_price <= (1 - stop_loss_threshold) * buy_price and not buy_signal.iloc[i]:
                stop_loss_amount = positions * current_price
                fee_paid = stop_loss_amount * trading_fee
                total_fees_paid += fee_paid + margin_fee
                capital += stop_loss_amount - fee_paid - margin_fee
                percent_change = (current_price - buy_price) / buy_price * 100
                if print_statements:
                    print(f"Sold   {positions} positions at ${current_price:.2f} on {date} (Stop-Loss), total amount: ${stop_loss_amount:.2f}, percent change: {percent_change:.2f}%, {(current_price - buy_price)*positions}")
                positions = 0
                buy_price = 0
                num_orders += 1
                sell_dates.append(date)
                sell_prices.append(current_price)
                holding_periods.append(holding_period)
                margin_fees_paid.append(margin_fee)
                stop_losses += 1
                stop_losses_by_year[date.year] = stop_losses_by_year.get(date.year, 0) + 1  # Update stop-loss count for the year
                settlement_dates.append(add_business_days(date, 2))

        if buy_signal.iloc[i] and capital > 0 and positions == 0 and all(d <= date for d in settlement_dates):
            buy_price = current_price
            if leverage <= 1:
                positions = math.floor(capital * leverage / buy_price)
            else:
                positions = math.floor(capital * leverage / buy_price) // 2 * 2
            fee_paid = positions * buy_price * trading_fee
            margin = capital
            capital -= positions * buy_price - fee_paid
            if print_statements:       
                print(f"Bought {positions} positions at ${buy_price:.2f} on {date}, total amount: ${positions * buy_price:.2f}")
            num_orders += 1
            buy_dates.append(date)
            buy_prices.append(buy_price)
            buy_time = date

        # Track portfolio value at the end of each day
        portfolio_values[date] = capital + positions * current_price

    final_value = capital + positions * option_data_all_dates['close'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Calculate annual growth rates and percentage changes
    annual_growth_rates = {}
    previous_year_value = initial_capital
    start_year = option_data_all_dates.index[0].year
    end_year = option_data_all_dates.index[-1].year
    for year in range(start_year, end_year + 1):
        last_business_day = get_last_business_day(year, option_data_all_dates.index)
        end_of_year_value = portfolio_values.get(last_business_day, None)
        if end_of_year_value is not None:
            annual_growth_rate = (end_of_year_value - previous_year_value) / previous_year_value * 100
            annual_growth_rates[year] = annual_growth_rate
            previous_year_value = end_of_year_value

    return {
        'final_value': final_value,
        'total_return': total_return,
        'total_fees_paid': total_fees_paid,
        'successful_trades': successful_trades,
        'stop_losses': stop_losses,
        'stop_losses_by_year': stop_losses_by_year,  # Return stop-losses by year
        'successful_trades_by_year': successful_trades_by_year,  # Return successful trades by year
        'average_holding_period': np.mean(holding_periods) if holding_periods else 0,
        'buy_dates': buy_dates,
        'buy_prices': buy_prices,
        'sell_dates': sell_dates,
        'sell_prices': sell_prices,
        'annual_growth_rates': annual_growth_rates,
        'params': {
            'leverage': leverage,
            'sell_threshold': sell_threshold,
            'stop_loss_threshold': stop_loss_threshold,
            'trading_fee': trading_fee,
            'margin_fee_annual_rate': margin_fee_annual_rate
        }
    }


plot_graphs = True
send_emails = False
print_statements = True

# Run optimization with different parameter sets
param_sets = [
    {'leverage': 1, 'sell_threshold': 0.05, 'stop_loss_threshold': 0.05, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.10, 'stop_loss_threshold': 0.05, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.15, 'stop_loss_threshold': 0.05, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.20, 'stop_loss_threshold': 0.05, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.25, 'stop_loss_threshold': 0.05, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.05, 'stop_loss_threshold': 0.10, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.10, 'stop_loss_threshold': 0.10, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.15, 'stop_loss_threshold': 0.10, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.20, 'stop_loss_threshold': 0.10, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.25, 'stop_loss_threshold': 0.10, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},  
    # {'leverage': 2, 'sell_threshold': 0.05, 'stop_loss_threshold': 0.15, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.10, 'stop_loss_threshold': 0.15, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},  
    # {'leverage': 2, 'sell_threshold': 0.15, 'stop_loss_threshold': 0.15, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.20, 'stop_loss_threshold': 0.15, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
    # {'leverage': 2, 'sell_threshold': 0.25, 'stop_loss_threshold': 0.15, 'trading_fee': 0.006, 'margin_fee_annual_rate': 0.15},
]

# stock_symbols = ['AMZN', 'GOOG', 'MSFT', 'BRK-B', 'TSLA', 'INTC']
# stock_symbols = ['TSLA']
stock_symbols = [    
    'TSLA',  # Tesla, Inc.
    'AMD',
    'NVDA',  # NVIDIA Corporation
    'BYND',
    # 'AMGN',  # Amgen Inc.
    # 'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'AMZN',  # Amazon.com, Inc.
    'META',
    # 'GOOGL', # Alphabet Inc. (Class A)
    'MRNA',
    'LULU',  # Lululemon Athletica Inc.
    'VSCO',  # Victoria's Secret & Co.

]


# Define the historical dates as strings
historical_dates = [
    '2021-07-26',  # example dates, replace with actual ones
    # '2024-03-13'
]

# end_date = '2020-01-06'
end_date = datetime.now().strftime('%Y-%m-%d')

# Convert date strings to DatetimeIndex
historical_dates_index = pd.to_datetime(historical_dates)


# Load historical data
for file_path in stock_symbols:

    # Iterate over each date in historical_dates_index
    for date in historical_dates_index:

        # Convert date strings to datetime objects
        end_date_str = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Compare the dates
        if date > end_date_str:
            print(f"Warning: Start Date {date} is later than End Date {end_date_str}")
            pass
        else:   
            # Download stock data
            data = yf.download(file_path)

        # Ensure the date is within the data range
        if date in data.index:
            option_data_all_dates = data.loc[date:end_date].copy()

            # option_data_all_dates = yf.download(file_path)[i:end_index]
            option_data_all_dates.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)

            # Parameters
            initial_capital = 500

            # Calculate daily returns
            ret = option_data_all_dates['close'].pct_change().dropna()

            # Define buy and sell signals
            percentile_5 = np.percentile(ret, 10)
            percentile_95 = np.percentile(ret, 90)

            sell_signal = ret <= percentile_5
            buy_signal = ret >= percentile_95

            results = []
            for params in param_sets:
                result = backtest_strategy(initial_capital, **params, print_statements=print_statements)
                results.append(result)

            last_buy_signal = buy_signal[-1] # Placeholder, should be replaced with actual logic
            last_sell_signal = sell_signal[-1] # Placeholder, should be replaced with actual logic

            today = pd.to_datetime(datetime.today().date())

            if buy_signal.index[-1] == today:
                last_price = option_data_all_dates['close'][-1]
                if last_buy_signal and results[0]['buy_dates'][-1] == today:
                    date_str = today.strftime('%Y-%m-%d')
                    print(f'\n\n----Buy Signal Today {file_path} {date_str} ----\n\n')
                    if send_emails:
                        send_email(f"Buy Signal Today - {file_path} - {date_str}", f"Buy {file_path} at ${last_price:.2f} on {date_str} \n If this email was not between 3:50pm-4:00pm EST, then invalid")

            if sell_signal.index[-1] == today:
                last_price = option_data_all_dates['close'][-1]
                if last_sell_signal and results[0]['sell_dates'][-1] == today:
                    date_str = today.strftime('%Y-%m-%d')
                    print(f'\n\n----Sell Signal Today {file_path} {date_str} ----\n\n')
                    if send_emails:
                        send_email(f"Sell Signal Today - {file_path} - {date_str}", f"Sell {file_path} at ${last_price:.2f} on {date_str} \n If this email was not between 3:50pm-4:00pm EST, then invalid")

            # Printing the results
            if print_statements:
                # Find the best parameter set based on total return
                best_result = max(results, key=lambda x: x['total_return'])
                print(f"\nBest params: {best_result['params']} \n")

                print("\nAnnual Growth Rates For:")
                for year, growth_rate in best_result['annual_growth_rates'].items():
                    print(f"{year}: {growth_rate:.2f}%")
                
                print("\nTrades by Year:")
                years = sorted(set(best_result['successful_trades_by_year'].keys()).union(best_result['stop_losses_by_year'].keys()))
                for year in years:
                    successful_trades = best_result['successful_trades_by_year'].get(year, 0)
                    stop_losses = best_result['stop_losses_by_year'].get(year, 0)
                    print(f"{year}: Successful Trades: {successful_trades}, Stop-Losses: {stop_losses}")

                # Print results and annual growth rates
                print(f"Final Portfolio Value: ${best_result['final_value']:.2f}")
                print(f"Total Return: {best_result['total_return']:.2f}%")
                print(f"Total Fees Paid: ${best_result['total_fees_paid']:.2f}")
                print(f"Number of Successful Trades: {best_result['successful_trades']}")
                print(f"Number of Stop-Loss Trades: {best_result['stop_losses']}")
                print(f"Average Holding Period: {best_result['average_holding_period']:.2f} days")
                

            # Plotting
            if plot_graphs:
                plt.figure(figsize=(14, 7))
                plt.plot(option_data_all_dates.index, option_data_all_dates['close'], label='Closing Price', color='blue')
                plt.scatter(best_result['buy_dates'], best_result['buy_prices'], marker='^', color='green', label='Buy Signal', s=100)
                plt.scatter(best_result['sell_dates'], best_result['sell_prices'], marker='v', color='red', label='Sell Signal', s=100)
                plt.title(f'Buy and Sell Signals on {file_path} Price Chart')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)
                plt.show()

        else:
            print(f'Date not in Index')


# Add these lines where you end the process
end_time = get_current_time()
print(f"Process ended at {end_time}")


# Prepare email details
subject = "Process Execution Details"
body = f"Successfully began run at {start_time} and finished at {end_time}"

# Send the email
if send_emails:
    send_email(subject, body)