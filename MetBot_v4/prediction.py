import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO
import io
import os
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

#Списки символов для ОСРок - первый для классов main, add, serial, второй для класса model
#Они нужны при загрузке весов моделей
main_add_char_serial_list = ['|', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0', ' ']
model_char_list = ['|', 'Y', 'V', 'T', 'S', 'R', 'O', 'N', 'M', 'L', 'K', 'I', 'G', 'F', 'E', 'D', 'B', 'A', ' ']

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

#класс ОСР
class OCR_Model(torch.nn.Module):
    def __init__(self, char_list, out_len=16):
        super().__init__()

        # Инициализация n_classes на основе char_list
        self.n_classes = len(char_list) + 1  # +1 для blank token

        self.m = torchvision.models.resnet34(pretrained=True)
        self.blocks = [torch.nn.Conv2d(3, 64, 7, 1, 3), self.m.bn1, self.m.relu, self.m.maxpool,
                      self.m.layer1, self.m.layer2, self.m.layer3]
        self.feature_extractor = torch.nn.Sequential(*self.blocks)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((512, out_len))
        self.bilstm1 = torch.nn.LSTM(512, 256, 2, dropout=0.15, batch_first=True, bidirectional=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, self.n_classes))

    def forward(self, x, return_x=False):
        feature = self.feature_extractor(x)
        b, c, h, w = feature.size()
        feature = feature.view(b, c * h, w)
        feature = self.avg_pool(feature)
        feature = feature.transpose(1, 2)
        out, (h_t1, c_t1) = self.bilstm1(feature)
        # out, _ = self.bilstm2(out, (h_t1, c_t1))
        #print(x.shape)
        out = self.classifier(out)

        return out

#Вспомогательные функции для определения результатов ОСР
def GreedyDecoder(output, char_list, blank_label=None, collapse_repeated=True):
    if blank_label is None:
        blank_label = len(char_list)

    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index == blank_label:  # Для blank_label не добавляем символ в расшифровку
                continue
            if collapse_repeated and j != 0 and index == args[j - 1]:
                continue
            if index < len(char_list):  # Проверяем, что индекс в пределах списка
                decode.append(char_list[index])
            else:
                decode.append("")  # Если индекс вне диапазона, добавляем пустую строку или специальный символ
        decodes.append(''.join(decode))
    return decodes

# картинку -> в чб
def black2white(image):

    lo=np.array([0,0,0])
    hi=np.array([0,0,0])
    mask = cv2.inRange(image, lo, hi)
    image[mask>0]=(255,255,255)

    return image


#Функция для распознавания текста основных показателей и показателей после запятой, класс main и класс add
#На вход путь к картинке и модель(ocr_main_model - для main, ocr_add_model - для add)
def process_main_add_number(image_path, ocr_model):
    # Load image
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = black2white(np.array(image))

    # Convert image to a format suitable for the model
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32) / 255
    image = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.FloatTensor(np.moveaxis(image, -1, 0)).unsqueeze(0).to(device)  # Add batch dimension

    # Process the image with the model
    with torch.no_grad():
        ocr_model.eval()
        basic_output = torch.nn.functional.log_softmax(ocr_model(image_tensor), dim=-1)
        preds = GreedyDecoder(basic_output, main_add_char_serial_list)

    # Get the recognized text
    decoded_text = preds[0].replace('|', " ").strip()
    return decoded_text

#Функция для распознавания названия модели счетчика, класс model
#На вход путь к картинке и модель(ocr_model_model)
def process_model_image(image_path, ocr_model):
    # Load image
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = black2white(np.array(image))

    # Convert image to a format suitable for the model
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32) / 255
    image = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.FloatTensor(np.moveaxis(image, -1, 0)).unsqueeze(0).to(device)  # Add batch dimension

    # Process the image with the model
    with torch.no_grad():
        ocr_model.eval()
        basic_output = torch.nn.functional.log_softmax(ocr_model(image_tensor), dim=-1)
        preds = GreedyDecoder(basic_output, model_char_list)

    # Get the recognized text
    decoded_text = preds[0].replace('|', " ").strip()
    if decoded_text in ('ALMES', 'ALLMES', 'ALMESS'):
        decoded_text = 'ALLMESS'  # Используйте присваивание, а не сравнение
    elif decoded_text == 'BF':
        decoded_text = 'BFG'
    return decoded_text

#Функция для распознавания серийного номера, класс serial
#На вход путь к картинке и модель(ocr_serial_model)
def process_serial_image(image_path, ocr_model):
    # Load image
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = black2white(np.array(image))

    # Convert image to a format suitable for the model
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32) / 255
    image = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.FloatTensor(np.moveaxis(image, -1, 0)).unsqueeze(0).to(device)  # Add batch dimension

    # Process the image with the model
    with torch.no_grad():
        ocr_model.eval()
        basic_output = torch.nn.functional.log_softmax(ocr_model(image_tensor), dim=-1)
        preds = GreedyDecoder(basic_output, main_add_char_serial_list)

    # Get the recognized text
    decoded_text = preds[0].replace('|', " ").strip()
    return decoded_text

#Функция для ЙОЛО, которая обрезает 4 картинки классов main, add, serial и model
#На вход путь к картинке и модель(crop_img_model)
def process_and_crop_image(model, image_path):
    # Выполняем детектирование объектов
    results = model(image_path)

    # Инициализация переменных для хранения путей к файлам каждого класса
    path_for_main = None
    path_for_add = None
    path_for_serial = None
    path_for_model = None

    for result_idx, result in enumerate(results):
        # Получаем изображение и его оригинальный размер
        orig_image = Image.fromarray(result.orig_img)
        orig_size = result.orig_shape

        for i, box in enumerate(result.boxes.xyxy):
            # Получаем класс объекта
            class_id = result.boxes.cls[i].int().item()
            class_name = result.names[class_id]

            # Проверяем, принадлежит ли объект к одному из интересующих классов
            if class_name in ["main", "add", "model", 'serial']:
                xmin, ymin, xmax, ymax = box[:4].int().tolist()
                cropped_img = orig_image.crop((xmin, ymin, xmax, ymax))

                # Сохраняем обрезанное изображение
                filename = f"cropped_{result_idx}_{i}_{class_name}.png"
                cropped_img.save(filename)

                # Сохраняем путь к файлу в соответствующую переменную
                if class_name == "main":
                    path_for_main = filename
                elif class_name == "add":
                    path_for_add = filename
                elif class_name == "model":
                    path_for_model = filename
                elif class_name == 'serial':
                    path_for_serial = filename

    return path_for_main, path_for_add, path_for_serial, path_for_model


#Подгрузка всех моделей и весов
ocr_main_model = OCR_Model(char_list=main_add_char_serial_list)
ocr_main_model.load_state_dict(torch.load('models/OCR_main_20_03.pt', map_location=torch.device('cpu')))  # основные показатели
ocr_main_model = ocr_main_model.to(device)

ocr_add_model = OCR_Model(char_list=main_add_char_serial_list)
ocr_add_model.load_state_dict(torch.load('models/OCR_add_23_03_2024.pt', map_location=torch.device('cpu')))  # показатели после запятой
ocr_add_model = ocr_add_model.to(device)

ocr_serial_model = OCR_Model(char_list=main_add_char_serial_list)
ocr_serial_model.load_state_dict(torch.load('models/OCR_serial_26_03_2024.pt', map_location=torch.device('cpu')))  # серийник
ocr_serial_model = ocr_serial_model.to(device)

ocr_model_model = OCR_Model(char_list=model_char_list)
ocr_model_model.load_state_dict(torch.load('models/OCR_model_26_03_2024.pt', map_location=torch.device('cpu')))  # название модели счетчика
ocr_model_model = ocr_model_model.to(device)

crop_img_model = YOLO('models/Yolov8_counters_4_classes.pt') # ЙОЛА которая вырезает нужные картинки из оригинального изображения

#Функция выдающая словарь со всеми результатами
def meters_prediction(image_path):
    # Crop the images
    path_for_main, path_for_add, path_for_serial, path_for_model = process_and_crop_image(crop_img_model, image_path)
    
    path_list = [path_for_main, path_for_add, path_for_serial, path_for_model]

    # Initialize a dictionary to hold the results
    results = {
        "Key digits": None,
        "Decimal digits": None,
        "Counter model name": None,
        "Serial number": None
    }
    results_rus = {
        "Основные показатели": None,
        "Показатели после запятой": None,
        "Название модели счетчика": None,
        "Серийный номер": None
    }

    # Process each cropped image and update the results dictionary
    if path_for_main:
        results["Key digits"] = process_main_add_number(path_for_main, ocr_main_model)
    if path_for_add:
        results["Decimal digits"] = process_main_add_number(path_for_add, ocr_add_model)
    if path_for_model:
        results["Counter model name"] = process_model_image(path_for_model, ocr_model_model)
    if path_for_serial:
        results["Serial number"] = process_serial_image(path_for_serial, ocr_serial_model)
        
    try:    
        for p in path_list:
            os.remove(p)
    except:
        print('File saving error!')
        
    new_keys = ["Digits", "Counter model name", "Serial number", "DateTime"]
    res = dict.fromkeys(new_keys)
    res["Digits"] = f"{results['Key digits']}.{results['Decimal digits']}"
    res["Counter model name"] = results["Counter model name"]
    res["Serial number"] = results["Serial number"]
    res["DateTime"] = datetime.now().strftime('%d.%m.%Y, %H:%M:%S')
    

    return res

# im_path = 'images/2.jpg'
# print(meters_prediction(im_path))
