import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def calculate_political_index(input_file):
    """Расчет индекса динамики политического процесса"""
    df = pd.read_csv(input_file)

    # Заполнение пропущенных GDP
    df['GDP'] = df['GDP'].replace(0, np.nan).fillna(df['Exports'] + df['Imports'])

    # Создание дополнительных показателей
    df['Trade_Balance'] = df['Exports'] - df['Imports']
    df['Exchange_Rate_Stability'] = (df['AMA_exchange_rate'] - df['IMF_exchange_rate']).abs()

    # Веса показателей
    indicators = {
        'GDP': 0.20,
        'Exports': 0.05,
        'Imports': 0.05,
        'Trade_Balance': 0.10,
        'Per_capita_GNI': 0.15,
        'Exchange_Rate_Stability': 0.10,
        'Household_consumption': 0.10,
        'Population': 0.05,
        'Government_expenditure': 0.10
    }

    # Нормализация данных
    scaler = MinMaxScaler()
    for indicator in indicators:
        df[f'{indicator}_norm'] = scaler.fit_transform(df[[indicator]])

    # Расчет индекса (0-100)
    df['Index'] = (sum(df[f'{ind}_norm'] * weight for ind, weight in indicators.items()) * 100).round(2)

    return df[['Country', 'Year', 'Index']]


def create_country_year_table(df):
    """Создает таблицу с индексами по годам для каждой страны"""
    pivot_table = df.pivot_table(
        index='Country',
        columns='Year',
        values='Index',
        aggfunc='first'
    ).reset_index()

    years = sorted([col for col in pivot_table.columns if isinstance(col, (int, float))])
    return pivot_table[['Country'] + years]


def plot_yearly_average(df):
    """Визуализация среднего индекса по годам"""
    yearly_avg = df.groupby('Year')['Index'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(yearly_avg['Year'], yearly_avg['Index'],
             marker='o', linestyle='-', linewidth=2, markersize=8, color='royalblue')

    plt.title('Средний индекс политической динамики по годам', fontsize=14)
    plt.xlabel('Год', fontsize=12)
    plt.ylabel('Средний индекс (0-100)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Значения на графике
    for x, y in zip(yearly_avg['Year'], yearly_avg['Index']):
        plt.text(x, y+1, f'{y:.1f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(yearly_avg['Year'], rotation=45)
    plt.tight_layout()
    plt.show()

    return yearly_avg


if __name__ == "__main__":
    input_file = "political_dynamics_index2.csv"
    output_file = "country_year_index_table.csv"

    # Расчет индексов
    print("Расчет индексов...")
    index_df = calculate_political_index(input_file)

    # Создание таблицы по странам и годам
    print("Формирование таблицы...")
    result_table = create_country_year_table(index_df)
    result_table.to_csv(output_file, index=False)
    print(f"\nТаблица сохранена в {output_file}")

    # Визуализация среднего индекса по годам
    print("\nВизуализация среднего индекса...")
    yearly_avg = plot_yearly_average(index_df)

    # Дополнительный вывод данных, для проверки работоспособности
    print("\nСредние значения индекса по годам:")
    print(yearly_avg)
