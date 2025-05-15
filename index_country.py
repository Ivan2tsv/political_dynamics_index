import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def calculate_comprehensive_index(input_file, output_file):
    """
    Рассчитывает расширенный индекс политической динамики на основе:
    - Экономических показателей (GDP, Exports, Imports)
    - Финансовых показателей (обменный курс, ВНД)
    - Социальных показателей (потребление, население)
    - Государственных показателей (расходы)
    """
    try:
        # Загрузка данных
        df = pd.read_csv(input_file)

        # Проверка обязательных колонок
        required_columns = [
            'Country', 'Year', 'GDP', 'Exports', 'Imports',
            'AMA_exchange_rate', 'IMF_exchange_rate',
            'Per_capita_GNI', 'Population',
            'Household_consumption', 'Government_expenditure'
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Предупреждение: отсутствуют колонки: {missing_cols}")
            print("Доступные колонки:", df.columns.tolist())
            return None

        # Очистка данных
        df = df.replace(0, np.nan)

        # Заполнение пропущенных значений GDP
        df['GDP'] = df['GDP'].fillna(df['Exports'] + df['Imports'])

        # Создание дополнительных показателей
        df['Trade_Balance'] = df['Exports'] - df['Imports']
        df['Exchange_Rate_Stability'] = (df['AMA_exchange_rate'] - df['IMF_exchange_rate']).abs()

        # Список всех показателей для индекса с весами
        indicators = {
            # Экономические показатели (40%)
            'GDP': 0.20,
            'Exports': 0.05,
            'Imports': 0.05,
            'Trade_Balance': 0.10,

            # Финансовые показатели (25%)
            'Per_capita_GNI': 0.15,
            'Exchange_Rate_Stability': 0.10,

            # Социальные показатели (20%)
            'Household_consumption': 0.10,
            'Population': 0.10,

            # Государственные показатели (15%)
            'Government_expenditure': 0.15
        }

        # Нормализация показателей
        scaler = MinMaxScaler()
        for indicator in indicators.keys():
            df[f'{indicator}_norm'] = scaler.fit_transform(df[[indicator]])

        # Расчет составного индекса
        df['Political_Dynamics_Index2'] = sum(
            df[f'{ind}_norm'] * weight
            for ind, weight in indicators.items()
        )

        # Приведение к шкале 0-100
        df['Political_Dynamics_Index'] = (df['Political_Dynamics_Index2'] * 100).round(2)

        # Создание итоговой таблицы
        result_cols = ['Country', 'Year', 'Political_Dynamics_Index2'] + list(indicators.keys())
        result_df = df[result_cols].sort_values(['Country', 'Year'])

        # Сохранение результатов
        result_df.to_csv(output_file, index=False)
        print(f"Расширенный индекс рассчитан и сохранен в {output_file}")

        return result_df

    except Exception as e:
        print(f"Ошибка при обработке данных: {str(e)}")
        return None


def visualize_index_trends(df, countries=None, years=None):
    """Визуализация динамики индекса для выбранных стран и лет"""
    if df is None:
        print("Нет данных для визуализации")
        return

    import matplotlib.pyplot as plt

    # Фильтрация данных
    if countries:
        df = df[df['Country'].isin(countries)]
    if years:
        df = df[df['Year'].isin(years)]

    plt.figure(figsize=(14, 7))

    for country in df['Country'].unique():
        country_data = df[df['Country'] == country]
        plt.plot(country_data['Year'], country_data['Political_Dynamics_Index2'],
                label=country, marker='o', linewidth=2)

    plt.title('Динамика мирового политического процесса', fontsize=14)
    plt.xlabel('Год', fontsize=12)
    plt.ylabel('Индекс (0-100)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# запуск программы
if __name__ == "__main__":
    input_csv = "political_dynamics_index2.csv"
    output_csv = "comprehensive_political_index.csv"

    print("Расчет комплексного индекса политической динамики...")
    results = calculate_comprehensive_index(input_csv, output_csv)

    if results is not None:
        print("\nПервые 5 строк результатов:")
        print(results.head())

        # Визуализация для выбранных стран
        selected_countries = ['United States', 'China', 'Germany', 'India', 'France', 'Japan', 'Russian Federation']
        visualize_index_trends(results, countries=selected_countries)
