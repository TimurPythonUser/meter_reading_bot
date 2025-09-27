import sqlite3
import os
import pandas as pd

def insert_data_to_sqlite(db_path, data):
    try:
        # Підключення до бази даних SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Створення таблиці, якщо вона не існує
        cursor.execute('''CREATE TABLE IF NOT EXISTS Counters
                          (UserID TEXT, Digits TEXT, CounterModelName TEXT, SerialNumber TEXT, DateTime TEXT, ImagePath TEXT)''')

        # Вставка даних зі словника в таблицю
        cursor.execute("INSERT INTO Counters VALUES (?, ?, ?, ?, ?, ?)",
                       (data.get('User ID'), 
                        data.get('Digits'), 
                        data.get('Counter model name'), 
                        data.get('Serial number'),
                        data.get('DateTime'),
                        data.get('ImagePath')
                        ))

        # Збереження змін в базі даних
        conn.commit()

        print("Данные успешно добавлены в базу данных")
    except Exception as e:
        print(f"Возникла ошибка: {e}")
    finally:
        # Закриття підключення до бази даних
        conn.close()


def append_to_excel(excel_path, res):
    # Проверка наличия файла Excel
    if not os.path.exists(excel_path):
        # Создание нового файла Excel, если он не существует
        df = pd.DataFrame(columns=['User ID', 'Digits', 'Counter model name', 'Serial number', 'DateTime', 'ImagePath'])
    else:
        # Загрузка существующего файла Excel
        df = pd.read_excel(excel_path, dtype=str)
    
    # Добавление новой строки в DataFrame
    df_res = pd.DataFrame([res], dtype=str)
    
    df = pd.concat([df, df_res], ignore_index=True)
    
    # Запись обновленного DataFrame в файл Excel
    df.to_excel(excel_path, index=False)
