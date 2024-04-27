import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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

def train_models(X_train, y_train, X_test, y_test):
    # Random Forest experiments
    rf_results = []
    for n_estimators in [50, 100, 200]:
        for max_depth in [None, 10, 20]:
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            report_rf = classification_report(y_test, y_pred_rf)
            rf_results.append(((n_estimators, max_depth), accuracy_rf, report_rf))

    # Logistic Regression experiments
    lr_results = []
    for C in [0.1, 1.0, 10.0]:
        for solver in ['lbfgs', 'liblinear', 'sag']:
            lr = LogisticRegression(C=C, solver=solver, max_iter=1000)
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            accuracy_lr = accuracy_score(y_test, y_pred_lr)
            report_lr = classification_report(y_test, y_pred_lr)
            lr_results.append(((C, solver), accuracy_lr, report_lr))

    # Support Vector Machine experiments
    svm_results = []
    for C in [0.1, 1.0, 10.0]:
        for kernel in ['linear', 'rbf', 'poly']:
            svm = SVC(C=C, kernel=kernel)
            svm.fit(X_train, y_train)
            y_pred_svm = svm.predict(X_test)
            accuracy_svm = accuracy_score(y_test, y_pred_svm)
            report_svm = classification_report(y_test, y_pred_svm)
            svm_results.append(((C, kernel), accuracy_svm, report_svm))

    # K Nearest Neighbors experiments
    knn_results = []
    for n_neighbors in [3, 5, 10]:
        for weights in ['uniform', 'distance']:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            knn.fit(X_train, y_train)
            y_pred_knn = knn.predict(X_test)
            accuracy_knn = accuracy_score(y_test, y_pred_knn)
            report_knn = classification_report(y_test, y_pred_knn)
            knn_results.append(((n_neighbors, weights), accuracy_knn, report_knn))

    return rf_results, lr_results, svm_results, knn_results


def print_results_table(results):
    for result in results:
        print(f"Model: {result[0]}")
        print(f"Accuracy: {result[1]}")
        print("Classification Report:")
        print(result[2])
        print()

def main():
    df = pd.read_csv('nflx_2014_2023.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle datetime separately if needed, or drop if not used
    # For example, extract year if relevant:
    # df['year'] = df['date'].dt.year

    pm = PriceMovement(df)
    pm.categorize()
    
    volm = Volume(df)
    volm.categorize()

    # Print the number of members in each class
    print("Class distribution in PriceMovement:")
    print(df['PriceMovement'].value_counts())

    print("Class distribution in Volume:")
    print(df['Volume'].value_counts())

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Date vs. Volume with color bar
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
    colors = {'Up': 'blue', 'Down': 'orange', 'Stable': 'green'}
    for category, color in colors.items():
        subset = df[df['PriceMovement'] == category]
        axes[1].scatter(subset['date'], subset['close'], label=category, color=color, alpha=0.6)
    axes[1].set_title('Date vs. Close Price')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Close Price')
    axes[1].legend(title='Price Movement')
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Price Movement vs. Volume
    colors = ['blue', 'orange', 'green']  # Color mapping for price movements
    categories = df['PriceMovement'].cat.categories
    for category, color in zip(categories, colors):
        subset = df[df['PriceMovement'] == category]
        axes[2].scatter(subset['Volume'].cat.codes, subset['volume'], label=category, color=color, alpha=0.6)
    axes[2].set_title('Price Movement vs. Volume')
    axes[2].set_xlabel('Price Movement Category')
    axes[2].set_ylabel('Volume')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Apply imputation
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)  # Ensure categorical and datetime data are also transformed if they were not excluded

    # Check again for NaN values
    if df.isnull().any().any():
        raise Exception("NaN values remain in the dataset after imputation.")

    # Prepare data for classification
    features = df.drop(columns=['Volume', 'PriceMovement', 'date'])  # Assuming 'date' is dropped unless features extracted
    target = df['Volume']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Train models and evaluate
    rf_results, lr_results, svm_results, knn_results = train_models(X_train, y_train, X_test, y_test)

    print("Random Forest Results:")
    print_results_table(rf_results)

    print("Logistic Regression Results:")
    print_results_table(lr_results)

    print("Support Vector Machine Results:")
    print_results_table(svm_results)

    print("K Nearest Neighbors Results:")
    print_results_table(knn_results)

if __name__ == "__main__":
    main()