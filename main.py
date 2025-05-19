import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Загружаем переменные окружения из файла .env
load_dotenv()

# Получаем API ключ из переменной окружения
google_api_key = os.getenv("GOOGLE_API_KEY")

# Проверяем, что ключ был загружен
if not google_api_key:
    print("Ошибка: API ключ Google не найден в переменных окружения.")
    print("Пожалуйста, убедитесь, что файл .env существует и содержит строку GOOGLE_API_KEY=...")
else:
    print("API ключ Google успешно загружен.")

    # 2. Инициализируем языковую модель Google Generative AI
    # Используем модель 'gemini-1.5-flash-latest', которая есть в списке доступных
    try:
        # Передаем google_api_key непосредственно в конструктор ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)
        print(f"Языковая модель {llm.model} инициализирована.")
    except Exception as e:
        print(f"Ошибка при инициализации модели: {e}")
        print("Убедитесь, что имя модели правильное и API ключ корректен.")
        exit() # Выходим, если не удалось инициализировать модель

    # 3. Отправляем тестовый запрос к модели
    user_query = "Привет! Расскажи мне что-нибудь интересное."
    print(f"\nОтправляем запрос к модели: '{user_query}'")

    try:
        # Создаем сообщение от пользователя
        messages = [HumanMessage(content=user_query)]

        # Отправляем запрос и получаем ответ
        response = llm.invoke(messages)

        # Выводим ответ
        print("\nОтвет модели:")
        print(response.content)

    except Exception as e:
        print(f"\nПроизошла ошибка при обращении к модели: {e}")
        print("Возможно, проблема с API ключом, доступом к модели или лимитами бесплатного уровня.")