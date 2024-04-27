import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Last inn CSV-filen som en DataFrame
df = pd.read_csv('nflx_2014_2023.csv')

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Vis grunnleggende deskriptiv statistikk
print(df.describe())

# Sjekk for manglende verdier
print(df.isnull().sum())

# Plott sluttkursen over tid
plt.figure()
df['close'].plot(title='Netflix Sluttkurs Over Tid')
plt.xlabel('Dato')
plt.ylabel('Sluttkurs')
plt.show()

# Beregn og plott et glidende gjennomsnitt for 50 dager av sluttkursenß
plt.figure()
df['sma_50'] = df['close'].rolling(window=50).mean()
df['close'].plot(label='Sluttkurs')
df['sma_50'].plot(label='50-dagers SMA', alpha=0.8)
plt.legend()
plt.show()

# Figur 3: Histogram for volum
plt.figure()
df['volume'].hist()
plt.title('Histogram av Volum')
plt.xlabel('Volum')
plt.ylabel('Frekvens')
plt.show()

# Finn og vis korrelasjoner
correlations = df.corr()
print(correlations)

# Visualiser fordelingene av ulike funksjoner med et histogramß
df.hist(figsize=(12, 10))
plt.show()

# Lag en scatter matrix for å se på potensielle forhold
scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.show()



# Deloppgave 5
# Identifiser manglende verdier
print(df.isnull().sum())

# Håndtere manglende verdier, eksempel ved å fylle med medianen for hver kolonne
first_valid_index = df['sma_50'].first_valid_index()
first_valid_value = df.loc[first_valid_index, 'sma_50']
df['sma_50'].fillna(first_valid_value, inplace=True)

# Identifiser og fjern duplikater
print(f"Antall duplikatrader før fjerning: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Antall duplikatrader etter fjerning: {df.duplicated().sum()}")



# Eksempel for å identifisere og håndtere outliers for 'close' kolonnen
Q1 = df['close'].quantile(0.25)
Q3 = df['close'].quantile(0.75)
IQR = Q3 - Q1

# Definer grenser for hva som betraktes som en outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifiser outliers
outliers = df[(df['close'] < lower_bound) | (df['close'] > upper_bound)]
print(f"Antall outliers i 'close': {len(outliers)}")

# Håndtere outliers, eksempel ved å fjerne dem
df = df[(df['close'] >= lower_bound) & (df['close'] <= upper_bound)]

