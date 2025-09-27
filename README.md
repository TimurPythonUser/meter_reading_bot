# 🤖 AI Meter Reader Bot 💡

Tired of squinting at tiny dials and manually typing utility meter readings? This intelligent Telegram bot leverages **AI-powered computer vision** to read and record water, gas, or electricity meter data directly from a photo.

Just send a picture, and let the AI do the work.



## ⭐ Key Features

-   🧠 **AI-Powered Recognition**: Utilizes a powerful **YOLOv8** model to detect key areas (readings, serial number, model) and a suite of custom **OCR models** to accurately transcribe the digits and text.

-   📸 **Effortless Workflow**: Simply snap a photo of your meter and send it to the bot. No apps to install, no complicated menus.

-   ✍️ **Intuitive Correction Flow**: If the AI misreads a digit, a simple and guided conversation flow (powered by `aiogram` FSM) allows you to quickly edit the data before saving.

-   💾 **Dual Data Storage**: Results are saved to both a robust **SQLite database** for structured data management and a user-friendly **Excel file**, perfect for easy viewing, sharing, or reporting.

-   🖼️ **Image Archiving**: Each reading is saved with the original photo, providing a visual record for verification and dispute resolution.

---

## 🚀 How It Works

The process is incredibly simple:

1.  **Send a Photo**: Open the chat with the bot and send a clear picture of your utility meter.
2.  **Get Instant Results**: The bot analyzes the image and sends back the recognized data in seconds.
3.  **Verify & Correct**: Review the data. If everything is correct, press "Save". If not, press "Edit" and follow the prompts.
4.  **Done!**: Your reading is securely saved in the database and the Excel log.

---

## 🛠️ Tech Stack

-   **Bot Framework**: [Aiogram 3](https://github.com/aiogram/aiogram)
-   **Object Detection**: [Ultralytics YOLOv8](https://ultralytics.com/)
-   **OCR & Backend**: [PyTorch](https://pytorch.org/), OpenCV
-   **Data Handling**: Pandas, Openpyxl, SQLite

---

## ⚙️ Setup & Installation

Follow these steps to get your own instance of the bot running.

### 1. Clone the Repository

```bash
git clone [https://github.com/TimurPythonUser/meter_reading_bot.git](https://github.com/TimurPythonUser/meter_reading_bot.git)
cd meter_reading_bot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Trained Models ❗️

The pre-trained AI models are essential for the bot to function. They are not stored in this repository due to their size.

**➡️ [Download the models from this Google Drive link](https://drive.google.com/drive/folders/1BvZyfA5IVk-2V6-WBNmGHQdZ53J34zeb?usp=drive_link)**

Create a folder named `models` in the main project directory and place all the downloaded `.pt` files inside it.

### 5. Set Up Environment Variables

Create a file named `.env` in the project root and add your Telegram bot token:

```env
TOKEN=YOUR_UNIQUE_TELEGRAM_BOT_TOKEN
```

### 6. Run the Bot

```bash
python aiogram_bot_buttons.py
```

Your bot is now live and ready to recognize meters!