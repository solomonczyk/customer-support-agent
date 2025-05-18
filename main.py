import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

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
    # 2. Инициализируем языковую модель
    # Мы используем ChatGoogleGenerativeAI для чатовых моделей Gemini
    # model="gemini-pro" указывает на конкретную модель (может потребоваться другая в зависимости от региона/доступа)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    print(f"Языковая модель {llm.model_name} инициализирована.")

    # 3. Отправляем простой запрос к модели
    # LangChain использует объекты "Message" для представления реплик в диалоге
    message = HumanMessage(content="Привет! Расскажи мне что-нибудь интересное.")

    print(f"\nОтправляем запрос к модели: '{message.content}'")

    try:
        # Вызываем модель и получаем ответ
        response = llm.invoke([message]) # invoke принимает список сообщений
        print("\nОтвет модели:")
        # Ответ модели также является объектом Message, content содержит текст
        print(response.content)

    except Exception as e:
        print(f"\nПроизошла ошибка при обращении к модели: {e}")
        print("Возможно, проблема с API ключом, доступом к модели или лимитами бесплатного уровня.")