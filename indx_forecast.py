import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Открытие данных
data = pd.read_csv('country_year_index_table.csv')

# Выбор страны
selected_countries = ['United States', 'China', 'Germany', 'India', 'France', 'Japan', 'Russian Federation']
data = data[data['Country'].isin(selected_countries)]

# Преобразование данных
df = data.melt(id_vars=['Country'], var_name='Year', value_name='Index')
df['Year'] = df['Year'].astype(float)


# Функция для прогнозирования с линейной регрессией
def forecast_index(country_data, years_to_forecast):
    X = country_data['Year'].values.reshape(-1, 1)
    y = country_data['Index'].values

    # Для России учитываем только годы с данными
    if country_data['Country'].iloc[0] == 'Russian Federation':
        country_data = country_data.dropna()
        X = country_data['Year'].values.reshape(-1, 1)
        y = country_data['Index'].values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(X[-1][0] + 1, X[-1][0] + years_to_forecast + 1).reshape(-1, 1)
    forecast = model.predict(future_years)

    # Для Японии ограничиваем рост (поскольку наблюдается стагнация)
    if country_data['Country'].iloc[0] == 'Japan':
        last_value = y[-1]
        forecast = np.array([last_value + 0.1 * i for i in range(1, years_to_forecast + 1)])

    # Для Китая добавляем экспоненциальный рост
    elif country_data['Country'].iloc[0] == 'China':
        last_value = y[-1]
        growth_rate = 1.06  # 6% ежегодный рост
        forecast = np.array([last_value * (growth_rate ** i) for i in range(1, years_to_forecast + 1)])

    # Для Индии ускоряющийся рост
    elif country_data['Country'].iloc[0] == 'India':
        last_value = y[-1]
        growth_rate = 1.08  # 8% ежегодный рост с ускорением
        forecast = np.array([last_value * (growth_rate ** (i * 1.02)) for i in range(1, years_to_forecast + 1)])

    return future_years.flatten(), forecast


# Прогноз на 30 лет
forecast_years = 30
forecast_data = []

for country in selected_countries:
    country_data = df[df['Country'] == country]
    years, forecast = forecast_index(country_data, forecast_years)

    for year, value in zip(years, forecast):
        forecast_data.append({'Country': country, 'Year': year, 'Index': value})

# Создание DataFrame с прогнозами
forecast_df = pd.DataFrame(forecast_data)

# Объединение исходных данных и прогнозов
full_df = pd.concat([df, forecast_df]).sort_values(by=['Country', 'Year'])

# Преобразование обратно в широкий формат для вывода
output_df = full_df.pivot(index='Country', columns='Year', values='Index').reset_index()
output_df.columns.name = None

# Сохранение в CSV
output_df.to_csv('country_year_index_forecast.csv', index=False)

# Расчет среднего годового индекса по странам
mean_index = full_df.groupby('Year')['Index'].mean().reset_index()
mean_index['Lower'] = mean_index['Index'] * 0.85
mean_index['Upper'] = mean_index['Index'] * 1.15

# Сохранение средних значений
mean_index.to_csv('mean_index_with_bounds.csv', index=False)

# Визуализация прогнозов
plt.figure(figsize=(14, 8))
for country in selected_countries:
    country_data = full_df[full_df['Country'] == country]
    plt.plot(country_data['Year'], country_data['Index'], label=country)

plt.title('Прогноз индексов по странам до 2051 года')
plt.xlabel('Год')
plt.ylabel('Индекс')
plt.legend()
plt.grid(True)
plt.savefig('country_index_forecast.png')
plt.show()

# Визуализация среднего индекса
plt.figure(figsize=(14, 6))
plt.plot(mean_index['Year'], mean_index['Index'], label='Средний индекс', color='black', linewidth=2)
plt.fill_between(mean_index['Year'], mean_index['Lower'], mean_index['Upper'], color='gray', alpha=0.3, label='±15% отклонение')

plt.title('Средний годовой индекс по странам с отклонением ±15%')
plt.xlabel('Год')
plt.ylabel('Индекс')
plt.legend()
plt.grid(True)
plt.savefig('mean_index_with_bounds.png')
plt.show()

print("Анализ завершен. Результаты сохранены в файлы:")
print("- country_year_index_forecast.csv - полные данные с прогнозами")
print("- mean_index_with_bounds.csv - средние значения с границами отклонения")
print("- country_index_forecast.png - график прогнозов по странам")
print("- mean_index_with_bounds.png - график среднего индекса с отклонениями")
