import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv('globaleconomyindicators.csv', skipinitialspace=True)

# Удаление лишних пробелов и кавычек в названиях столбцов
df.columns = [col.strip().strip('"').strip() for col in df.columns]

# Разделение данных
# Получение столбцов из первой строки файла
all_columns = df.columns[0].split(',')
all_columns = [col.strip().strip('"').strip() for col in all_columns]

# Чтение файла без первой строки и заголовков
df = pd.read_csv('globaleconomyindicators.csv', skiprows=1, header=None, names=all_columns)

# Форматирование данных
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Преобразование числовых столбцов
numeric_cols = df.columns.difference(['Country', 'Currency'])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Выборка необходимых столбцов
selected_columns = [
    'Country', 'Year',
    'Gross Domestic Product (GDP)',
    'Gross National Income(GNI) in USD',
    'Exports of goods and services',
    'Imports of goods and services',
    'AMA exchange rate',
    'IMF based exchange rate',
    'Per capita GNI',
    'Population',
    'Household consumption expenditure (including Non-profit institutions serving households)',
    'Agriculture, hunting, forestry, fishing (ISIC A-B)',
    'Manufacturing (ISIC D)',
    'Construction (ISIC F)',
    'Transport, storage and communication (ISIC I)',
    'General government final consumption expenditure',
    'Gross capital formation'
]

# Проверка доступных столбцов
available_columns = [col for col in selected_columns if col in df.columns]
missing_columns = [col for col in selected_columns if col not in df.columns]

if missing_columns:
    print(f"Предупреждение: следующие столбцы не найдены в данных: {missing_columns}")

# Создание нового DataFrame
result_df = df[available_columns].copy()

# Замена пропущенных значений на 0
result_df = result_df.fillna(0)

# Переименование столбцов
column_rename = {
    'Gross Domestic Product (GDP)': 'GDP',
    'Gross National Income(GNI) in USD': 'GNI_USD',
    'Exports of goods and services': 'Exports',
    'Imports of goods and services': 'Imports',
    'AMA exchange rate': 'AMA_exchange_rate',
    'IMF based exchange rate': 'IMF_exchange_rate',
    'Per capita GNI': 'Per_capita_GNI',
    'Household consumption expenditure (including Non-profit institutions serving households)': 'Household_consumption',
    'Agriculture, hunting, forestry, fishing (ISIC A-B)': 'Agriculture',
    'Manufacturing (ISIC D)': 'Manufacturing',
    'Construction (ISIC F)': 'Construction',
    'Transport, storage and communication (ISIC I)': 'Transport_infrastructure',
    'General government final consumption expenditure': 'Government_expenditure',
    'Gross capital formation': 'Gross_capital_formation'
}

result_df = result_df.rename(columns=column_rename)

# Добавление торгового баланса
if 'Exports' in result_df.columns and 'Imports' in result_df.columns:
    result_df['Trade_balance'] = result_df['Exports'] - result_df['Imports']

# Сохранение в новый файл
result_df.to_csv('political_dynamics_index2.csv', index=False)

# Вывод информации
print("\nФайл 'political_dynamics_index2.csv' успешно создан!")
print(f"Количество строк: {len(result_df)}")
print(f"Количество стран: {result_df['Country'].nunique()}")
print(f"Годы: от {result_df['Year'].min()} до {result_df['Year'].max()}")
print("\nПервые 5 строк полученного датасета:")
print(result_df.head())
