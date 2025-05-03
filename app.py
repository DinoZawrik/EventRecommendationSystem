# app_friendly.py
import gradio as gr
import pandas as pd
import logging
import os
import time

# ... (все импорты и функции остаются такими же) ...
# Импортируем функции из проекта
from src.utils import load_config, load_object, setup_logging
from src.data_loader import load_datasets
from src.feature_engineering import create_features, preprocess_datetime
try:
    from src.utils import ModelType
except ImportError:
    from typing import Any as ModelType

# --- Глобальные переменные ---
CONFIG_PATH = "config/params.yaml"
config = None
model = None
processed_data_cache = None
events_data = None
example_user_ids = None
last_processed_time = 0

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Загружаем конфиг СРАЗУ ---
try:
    logging.info("Загрузка конфигурации при старте...")
    config = load_config(CONFIG_PATH)
    logging.info("Конфигурация успешно загружена.")
except Exception as e:
    logging.critical(f"Не удалось загрузить конфигурацию {CONFIG_PATH}: {e}")
    exit()

# --- Функция загрузки примерных User ID ---
def load_example_user_ids():
    global example_user_ids, config
    if example_user_ids is None:
        logging.info("Загрузка User ID из test.csv для примеров...")
        try:
            test_path = os.path.join(config['data']['raw_dir'], config['data']['test_csv'])
            test_df_ids = pd.read_csv(test_path, usecols=[config['features']['user_id_col']])
            example_user_ids = sorted(test_df_ids[config['features']['user_id_col']].unique().tolist())
            logging.info(f"Загружено {len(example_user_ids)} уникальных User ID.")
            MAX_EXAMPLES = 1000
            if len(example_user_ids) > MAX_EXAMPLES:
                 logging.warning(f"Ограничиваем примеры ID до {MAX_EXAMPLES}.")
                 example_user_ids = example_user_ids[:MAX_EXAMPLES]
        except FileNotFoundError:
            logging.error(f"Файл test.csv не найден: {test_path}")
            example_user_ids = []
        except Exception as e:
            logging.error(f"Ошибка при загрузке User ID: {e}")
            example_user_ids = []
    return example_user_ids

# --- Остальные функции загрузки (load_event_details, load_resources, get_processed_data) ---
def load_event_details():
    global events_data, config
    if events_data is None:
        logging.info("Загрузка данных о событиях (events.csv)...")
        try:
            events_path = os.path.join(config['data']['raw_dir'], config['data']['events_csv'])
            compression = 'gzip' if config['data']['events_csv'].endswith('.gz') else None
            cols_to_load = ['event_id', 'start_time', 'city', 'state', 'country']
            events_data = pd.read_csv(events_path, compression=compression, usecols=cols_to_load, dtype={'event_id': str})
            events_data['start_time'] = preprocess_datetime(events_data, 'start_time')
            events_data.set_index('event_id', inplace=True)
            logging.info(f"Данные о {len(events_data)} событиях загружены.")
        except FileNotFoundError:
            logging.error(f"Файл events.csv не найден: {events_path}")
            events_data = pd.DataFrame()
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных о событиях: {e}")
            events_data = pd.DataFrame()
    return events_data

def load_resources():
    global model, config
    load_event_details()
    if model is None:
        model_path = os.path.join(config['output']['model_dir'], config['output']['model_name'])
        logging.info(f"Загрузка модели из {model_path}...")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Модель не найдена: {model_path}.")
        model = load_object(model_path)
        logging.info("Модель успешно загружена.")

def get_processed_data(force_reload=False):
    global processed_data_cache, last_processed_time, config
    current_time = time.time()
    cache_ttl = 3600
    if force_reload or processed_data_cache is None or (current_time - last_processed_time > cache_ttl):
        logging.info("Обновление кэша обработанных данных...")
        raw_datasets = load_datasets(config)
        required_keys = ['users', 'events', 'event_attendees', 'test']
        if not all(key in raw_datasets for key in required_keys):
            missing = [key for key in required_keys if key not in raw_datasets]
            raise ValueError(f"Отсутствуют сырые данные: {missing}")
        start_fe = time.time()
        processed_datasets = create_features(raw_datasets, config)
        end_fe = time.time()
        logging.info(f"Генерация признаков завершена за {end_fe - start_fe:.2f} сек.")
        if 'test' not in processed_datasets: raise KeyError("Ключ 'test' отсутствует.")
        processed_data_cache = processed_datasets['test'].copy()
        last_processed_time = current_time
        logging.info(f"Данные для предсказания (test.csv) подготовлены.")
    else:
        logging.info("Использование кэшированных обработанных данных.")
    user_col = config['features']['user_id_col']
    event_col = config['features']['event_id_col']
    if user_col not in processed_data_cache.columns or event_col not in processed_data_cache.columns:
         raise KeyError(f"Колонки ID '{user_col}' или '{event_col}' отсутствуют.")
    return processed_data_cache

# --- Основная функция для Gradio (без изменений) ---
def get_recommendations_for_gradio(user_id_input, progress=gr.Progress(track_tqdm=True)):
    global config, model, events_data
    try:
        progress(0, desc="Проверка User ID...")
        if user_id_input is None or user_id_input == '': return "Ошибка: Выберите или введите User ID."
        try: user_id = int(user_id_input)
        except (ValueError, TypeError): return f"Ошибка: Введите корректный User ID."

        progress(0.1, desc="Подготовка данных...")
        prediction_df = get_processed_data()

        progress(0.3, desc="Поиск данных...")
        user_col = config['features']['user_id_col']
        event_col = config['features']['event_id_col']
        user_data = prediction_df[prediction_df[user_col] == user_id]
        if user_data.empty: return f"Для User ID {user_id} не найдено событий."

        progress(0.5, desc="Подготовка признаков...")
        selected_features = config['features']['selected_features']
        missing_features = [f for f in selected_features if f not in user_data.columns]
        if missing_features: return f"Ошибка: Отсутствуют признаки {missing_features}."
        X_user = user_data[selected_features].fillna(0)

        progress(0.7, desc="Предсказание...")
        probabilities = model.predict_proba(X_user)[:, 1]

        progress(0.8, desc="Ранжирование...")
        top_n = config['recommend']['top_n']
        recommendation_df = pd.DataFrame({
            event_col: user_data[event_col].astype(str),
            'probability': probabilities
        }).sort_values(by='probability', ascending=False).head(top_n)
        recommended_event_ids = recommendation_df[event_col].tolist()

        progress(0.9, desc="Форматирование...")
        if not recommended_event_ids: output_message = f"Нет рекомендаций для User ID {user_id}."
        else:
            output_lines = []
            if events_data is None or events_data.empty:
                logging.warning("Данные о событиях не загружены.")
                output_lines = [f"{i+1}. Event ID: {event_id}" for i, event_id in enumerate(recommended_event_ids)]
            else:
                for i, event_id in enumerate(recommended_event_ids):
                    if event_id in events_data.index:
                        event_details = events_data.loc[event_id]
                        city=event_details.get('city'); state=event_details.get('state')
                        city_is_nan=pd.isna(city); state_is_nan=pd.isna(state)
                        if city_is_nan and state_is_nan: location = "Место не указано"
                        elif state_is_nan: location = str(city)
                        elif city_is_nan: location = str(state)
                        else: location = f"{str(city)}, {str(state)}"
                        start_time_str = event_details['start_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(event_details['start_time']) else 'N/A'
                        line = f"{i+1}. ID: {event_id}\n   Место: {location}\n   Начало: {start_time_str}"
                        output_lines.append(line)
                    else: output_lines.append(f"{i+1}. Event ID: {event_id} (детали не найдены)")
            output_message = "\n\n".join(output_lines)
        progress(1, desc="Готово!")
        return output_message
    except FileNotFoundError as e: return f"Ошибка: Не найдены файлы модели. {e}"
    except KeyError as e: return f"Ошибка конфигурации/данных: {e}."
    except Exception as e:
        logging.exception("Непредвиденная ошибка.")
        return f"Произошла внутренняя ошибка: {e}"

# --- Создание интерфейса с gr.Blocks и CSS для ширины ---
example_ids = load_example_user_ids()

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 960px !important; margin: auto !important; }"
) as demo:

    # БЛОК С ЛОГОТИПОМ УДАЛЕН

    gr.Markdown(
        """
        # Система Рекомендации Событий
        Выберите User ID из списка или введите свой, чтобы получить топ-5 рекомендованных событий.
        *Примечание: Список содержит ID пользователей из тестового набора данных.*
        """
    )

    # Горизонтальный макет с колонками
    with gr.Row():
        with gr.Column(scale=1): # Левая колонка
            user_id_input = gr.Dropdown(
                label="Выберите или введите User ID",
                choices=example_ids,
                allow_custom_value=True,
                filterable=True
            )
            submit_button = gr.Button("Получить Рекомендации", variant="primary")

        with gr.Column(scale=2): # Правая колонка
            output_textbox = gr.Textbox(
                label="Рекомендованные События",
                lines=15,
                interactive=False
            )

    submit_button.click(
        fn=get_recommendations_for_gradio,
        inputs=[user_id_input],
        outputs=[output_textbox]
    )

# --- Запуск приложения ---
if __name__ == "__main__":
    logging.info("Запуск Gradio приложения...")
    try:
        load_resources()
        demo.launch()
    except FileNotFoundError as e: print(f"Критическая ошибка: {e}.")
    except Exception as e: print(f"Критическая ошибка: {e}")