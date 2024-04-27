import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pandas.plotting import scatter_matrix

class PriceMovement:
    def __init__(self, data):
        self.data = data

    def categorize(self):
        self.data['PriceChange'] = self.data['close'].diff()
        conditions = [
        (self.data['PriceChange'] > 0.5),
        (self.data['PriceChange'] < -0.5),
        (self.data['PriceChange'] >= -0.5) & (self.data['PriceChange'] <= 0.5)
        ]
        choices = ['Up', 'Down', 'Stable']
        self.data['PriceMovement'] = np.select(conditions, choices)
        self.data['PriceMovement'] = pd.Categorical(self.data['PriceMovement'], categories=choices, ordered=True)

    def plot(self):
        self.data['PriceMovement'].value_counts().plot(kind='bar')
        plt.title('Price Movement Categories')
        plt.xlabel('Price Movement')
        plt.ylabel('Number of Days')
        plt.show()

class Volume:
    def __init__(self, data):
        self.data = data

    def categorize(self):
        conditions = [
            (self.data['volume'] <= self.data['volume'].quantile(0.33)),
            (self.data['volume'] <= self.data['volume'].quantile(0.67)),
            (self.data['volume'] > self.data['volume'].quantile(0.67))
        ]
        choices = ['Low', 'Medium', 'High']
        self.data['Volume'] = pd.cut(self.data['volume'], bins=[0] + list(self.data['volume'].quantile([0.33, 0.67, 1.0])), labels=choices, include_lowest=True)

    def plot(self):
        self.data['Volume'].value_counts().plot(kind='bar')
        plt.title('Volume Levels')
        plt.xlabel('Volume')
        plt.ylabel('Number of Days')
        plt.show()

def main():
    df = pd.read_csv('nflx_2014_2023.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    pm = PriceMovement(df)
    pm.categorize()
    
    volm = Volume(df)
    volm.categorize()

    # Opprett subplotter
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Date vs. Volume med fargebar
    scatter = axes[0].scatter(df['date'], df['volume'], c=df['Volume'].cat.codes, cmap='viridis', alpha=0.6)
    axes[0].set_title('Date vs. Volume')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Volume')
    axes[0].set_yscale('log')
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    colorbar = fig.colorbar(scatter, ax=axes[0])
    colorbar.set_label('Volume Category')
    colorbar.set_ticks([0, 1, 2])
    colorbar.set_ticklabels(['Low', 'Medium', 'High'])
    
    # Plot 2: Date vs. Close Price by Price Movement
    colors = {'Up': 'blue', 'Down': 'orange', 'Stable': 'green'}  # Definerer farger for hver kategori
    for category, color in colors.items():
        subset = df[df['PriceMovement'] == category]
        axes[1].scatter(subset['date'], subset['close'], label=category, color=color, alpha=0.6)
    axes[1].set_title('Date vs. Close Price by Price Movement')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Close Price')
    axes[1].legend(title='Price Movement')  # Legger til en legend
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    
    # Plot Price Movement vs. Volume
    categories = df['PriceMovement'].cat.categories
    colors = ['blue', 'orange', 'green']
    for category, color in zip(categories, colors):
        subset = df[df['PriceMovement'] == category]
        axes[2].scatter(subset['Volume'].cat.codes, subset['volume'], label=category, color=color, alpha=0.6)
    axes[2].set_title('Price Movement vs. Volume')
    axes[2].set_xlabel('Price Movement Category')
    axes[2].set_ylabel('Volume')
    axes[2].legend()


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()