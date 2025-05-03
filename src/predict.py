# src/predict.py
import pandas as pd
import logging
import os
from typing import Dict, Any

# Убедимся, что импорт ModelType работает (хотя ошибка сейчас не в нем)
try:
    from src.utils import ModelType
except ImportError:
    logging.error("Не удалось импортировать ModelType из src.utils. Использую Any.")
    from typing import Any as ModelType # Временная заглушка на случай проблем с utils

# <<<--- ПРОВЕРЬ ОПРЕДЕЛЕНИЕ ФУНКЦИИ ОЧЕНЬ ВНИМАТЕЛЬНО --->>>
def generate_predictions(model: ModelType, X_test: pd.DataFrame) -> pd.Series:
    """Генерирует предсказания вероятностей для тестового набора."""
    logging.info(f"Генерация предсказаний для {len(X_test)} тестовых образцов...")

    # Базовая проверка на NaN перед предсказанием
    if X_test.isnull().any().any():
        logging.warning(f"Обнаружены NaN в тестовых признаках перед предсказанием: \n{X_test.isnull().sum()[X_test.isnull().sum() > 0]}")
        # В идеале, обработка пропусков должна происходить консистентно на этапе feature engineering
        # Если NaN остались, примени безопасный метод (заполнение 0 или медианой)
        # Пример: X_test = X_test.fillna(0) # Или используй предрасчитанную медиану
        logging.warning("Попытка предсказания несмотря на NaN. Рассмотрите улучшение обработки пропусков.")

    try:
        # Убедись, что модель ожидает столбцы, которые есть в X_test
        # Пример проверки (для LGBM): model_features = model.feature_name_
        # if set(model_features) != set(X_test.columns):
        #    logging.warning("Несовпадение признаков между моделью и тестовыми данными!")

        probabilities = model.predict_proba(X_test)[:, 1]
        logging.info("Предсказания успешно сгенерированы.")
        # Возвращаем как Series с тем же индексом, что и у X_test, для легкого выравнивания
        return pd.Series(probabilities, index=X_test.index)
    except ValueError as ve:
         logging.error(f"ValueError во время предсказания, часто из-за NaN или несовпадения признаков: {ve}")
         logging.error(f"Количество NaN в X_test:\n{X_test.isnull().sum()}")
         raise
    except Exception as e:
        logging.error(f"Ошибка во время генерации предсказаний: {e}")
        raise
# <<<--- КОНЕЦ ПРОВЕРКИ ФУНКЦИИ --->>>

def create_submission_file(
    processed_test_df: pd.DataFrame, # Тестовый DF *с* ID пользователя/события
    probabilities: pd.Series,
    config: Dict[str, Any]
    ):
    """Создает файл submission для Kaggle."""
    cfg_pred = config['predict']
    cfg_feat = config['features']
    cfg_output = config['output']
    user_col = cfg_feat['user_id_col']
    event_col = cfg_feat['event_id_col']
    # Исправленный путь: сохраняем submission в корневой директории проекта, как в исходном ноутбуке
    # submission_path = os.path.join(cfg_output['reports_dir'], cfg_pred['submission_file']) # Старый путь
    submission_path = cfg_pred['submission_file'] # Сохраняет в корень как 'submission.csv'

    logging.info("Создание файла submission...")

    # Выравниваем вероятности с processed_test_df по индексу
    if not processed_test_df.index.equals(probabilities.index):
         logging.warning("Несовпадение индексов между тестовыми данными и вероятностями. Попытка переиндексации.")
         # Попытка выровнять по индексу, предполагая, что он сохранился
         probabilities = probabilities.reindex(processed_test_df.index)
         if probabilities.isnull().any():
              logging.error("Не удалось надежно выровнять вероятности с тестовыми данными после переиндексации.")
              raise ValueError("Несовпадение данных для submission.")

    submission_df = processed_test_df[[user_col, event_col]].copy()
    submission_df['Probability'] = probabilities.values # Присваиваем выровненные вероятности

    # Ранжируем события для каждого пользователя
    logging.info("Ранжирование событий для каждого пользователя...")
    # Сортируем по User (возр), затем Probability (убыв) для подготовки к группировке
    submission_df = submission_df.sort_values(by=[user_col, 'Probability'], ascending=[True, False])

    # Группируем по пользователю и объединяем ID событий в строку через пробел
    # Убедимся, что события являются строками перед объединением
    ranked_events = submission_df.groupby(user_col)[event_col].apply(lambda x: ' '.join(x.astype(str)))

    # Формируем итоговый DataFrame для submission
    final_submission = pd.DataFrame(ranked_events).reset_index()
    final_submission.columns = ['User', 'Events'] # Формат Kaggle
    final_submission['User'] = final_submission['User'].astype(int)

    # Сохраняем файл submission
    try:
        # os.makedirs не нужен, если сохраняем в корневой директории
        # os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        final_submission.to_csv(submission_path, index=False)
        logging.info(f"Файл submission успешно сохранен в {submission_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении файла submission: {e}")
        raise