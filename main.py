import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

BRIDGE_1 = "fitbit_data/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/"


def print_colored(text, color):
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }
    print(f"{colors[color]}{text}{colors['reset']}")


def process_activity_date(df: DataFrame) -> DataFrame:
    # Change to date time object to work better with date analysis

    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    return df


def avg_daily_steps_analysis(df) -> DataFrame:
    # Add activity level columns to determine if person is active or not

    average_steps = df['TotalSteps'].mean()
    df['ActivityLevel'] = df['TotalSteps'].apply(lambda x: 'Active' if x >= average_steps else 'Inactive')
    return df


def correlation_analysis(df):
    # Get strength and direction of linear relationship between total steps and calories burned

    correlation = df['TotalSteps'].corr(df['Calories'])
    print(f"Correlation between Total Steps and Calories Burned: {correlation}")


def get_most_active_day(df):
    most_active_day = df[df['VeryActiveMinutes'] == df['VeryActiveMinutes'].max()]
    print("MOST ACTIVE DAY:")
    print(most_active_day)


def get_daily_activity_data() -> DataFrame:
    data = BRIDGE_1 + "dailyActivity_merged.csv"
    df = pd.read_csv(data)
    return df


def get_data() -> DataFrame:
    df = get_daily_activity_data()
    return df


def visualize(df):
    # Plot Total Steps over Time

    plt.figure(figsize=(12, 6))
    plt.plot(df['ActivityDate'], df['TotalSteps'], label='Total Steps')
    plt.axhline(y=df['TotalSteps'].mean(), color='r', linestyle='--', label='Average Steps')
    plt.xlabel('Date')
    plt.ylabel('Total Steps')
    plt.title('Total Steps Over Time')
    plt.legend()
    plt.show()


def main():
    df = get_data()
    df = process_activity_date(df)
    df = avg_daily_steps_analysis(df)
    correlation_analysis(df)
    get_most_active_day(df)
    visualize(df)


main()
