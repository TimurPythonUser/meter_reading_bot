from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import StateFilter
from dotenv import load_dotenv
from PIL import Image
from prediction import meters_prediction
from database import insert_data_to_sqlite, append_to_excel
import numpy as np
import os
import asyncio
from aiogram.enums import ParseMode
import logging
logging.basicConfig(level=logging.INFO)
# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv('TOKEN')

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot=bot)

# Определение состояний для FSM
class MeterInput(StatesGroup):
    reading = State()
    serial_number = State()
    model_name = State()


# Регистрация команды /start
@dp.message(CommandStart())
async def command_start_handler(message: Message):
    await message.answer("Добрый день! Отправьте этому боту фотографию счетчика для распознавания показаний.")


@dp.message(F.photo)
async def photo_handler(message: Message, state: FSMContext):
    await message.answer("Мы получили от Вас фотографию. Идет распознавание показаний...")

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_path = file_info.file_path
    file = await bot.download_file(file_path)

    img = np.asarray(Image.open(file).convert('RGB'))
    result = {
        'User ID': message.from_user.id,
        'img': img  # Сохраняем изображение для последующего использования
    }
    result.update(meters_prediction(img))

    # Создаем копию данных для вывода, исключая 'img'
    display_result = {key: value for key, value in result.items() if key != 'img'}

    # Формируем строку результатов для отображения пользователю
    res_string = '\n'.join(f'{key}: {value}' for key, value in display_result.items())

    # Создаем кнопки и сразу же добавляем в InlineKeyboardMarkup
    save_button = InlineKeyboardButton(text="Сохранить", callback_data="save")
    edit_button = InlineKeyboardButton(text="Редактировать", callback_data="edit")
    markup = InlineKeyboardMarkup(inline_keyboard=[
        [save_button],
        [edit_button]
    ])

    # Сохраняем все необходимые данные в состояние
    await state.update_data(result)

    # Отправляем пользователю результаты с клавиатурой
    await message.answer(res_string, reply_markup=markup)


@dp.callback_query(lambda c: c.data == 'save')
async def process_save(callback_query: types.CallbackQuery, state: FSMContext):
    # Извлечение всех данных, сохраненных в состояние
    data = await state.get_data()
    img = data.get('img')  # Убедитесь, что 'img' был сохранен ранее

    if img is None:
        await callback_query.message.answer("Ошибка: изображение не найдено.")
        await state.clear()
        return

    # Вызов функции сохранения результатов
    save_results(data, img)
    await callback_query.message.answer("Информация сохранена! Можно продолжать работу!\n"
                                        "Отправьте этому боту фотографию счетчика для распознавания показаний.")

    # Очистка состояния после сохранения данных
    await state.clear()


@dp.callback_query(lambda c: c.data == "edit")
async def process_edit(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите показания прибора:")
    # Устанавливаем состояние с использованием FSMContext
    await state.set_state(MeterInput.reading)


@dp.message(StateFilter(MeterInput.reading))
async def input_reading(message: types.Message, state: FSMContext):
    # Сохраняем данные в контекст состояния
    await state.update_data(reading=message.text)
    await message.answer("Введите серийный номер:")
    await state.set_state(MeterInput.serial_number)


@dp.message(StateFilter(MeterInput.serial_number))
async def input_serial_number(message: types.Message, state: FSMContext):
    # Обновляем данные в контексте состояния
    await state.update_data(serial_number=message.text)
    await message.answer("Введите название модели:")
    await state.set_state(MeterInput.model_name)


@dp.message(StateFilter(MeterInput.model_name))
async def input_model_name(message: types.Message, state: FSMContext):
    # Извлекаем данные, включая изображение
    data = await state.get_data()
    img = data.get('img')  # Получаем изображение из состояния

    if img is None:
        await message.answer("Произошла ошибка при обработке изображения.")
        return

    await state.update_data(model_name=message.text)

    save_results(data, img)
    await message.answer("Информация обновлена и сохранена! Можно продолжать работу!\n"
                         "Отправьте этому боту фотографию счетчика для распознавания показаний.")
    await state.clear()


def save_results(data, img):
    user_id = data.get('User ID', 'Unknown User')
    # Учитываем возможность разных ключей для разных методов ввода
    digits = data.get('reading', data.get('Digits', 'Unknown'))
    serial_number = data.get('serial_number', data.get('Serial number', 'Unknown'))
    model_name = data.get('model_name', data.get('Counter model name', 'Unknown'))

    file_name = f"{user_id}_{digits.replace('.', '_')}.jpg"
    if not os.path.exists('Images'):
        os.makedirs('Images')
    save_path = os.path.join('Images', file_name)

    Image.fromarray(img).save(save_path)

    result = {
        'User ID': user_id,
        'Digits': digits,
        'Serial Number': serial_number,
        'Model Name': model_name,
        'ImagePath': save_path
    }
    db_path = 'MBotBase.db'
    insert_data_to_sqlite(db_path, result)
    excel_path = 'meters_data.xlsx'
    append_to_excel(excel_path, result)



async def runner():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(runner())


