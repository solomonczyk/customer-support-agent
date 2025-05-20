# requirements.txt:
# python-dotenv==1.0.0
# langchain==0.1.11
# langchain-core==0.1.27
# langchain-google-genai==0.0.10
# google-generativeai==0.3.2

import os
from dotenv import load_dotenv
import logging
# Импорты для LangChain и Google Gemini
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
except ImportError as e:
    print(f"ОШИБКА: {e}")
    print("\nДля работы скрипта требуются следующие библиотеки:")
    print("pip install python-dotenv langchain langchain-core langchain-google-genai google-generativeai")
    exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_api_key():
    """Загрузка API ключа из .env файла."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("API ключ Google не найден. Убедитесь, что GOOGLE_API_KEY установлен в файле .env.")
        return None
    logger.info("API ключ Google успешно загружен.")
    return api_key

def initialize_llm(api_key):
    """Инициализация языковой модели."""
    genai.configure(api_key=api_key)
    
    try:
        models = genai.list_models()
        available_models = []
        
        logger.info("Доступные модели Google Gemini:")
        for model in models:
            model_name = model.name
            logger.info(f" - {model_name}")
            name_parts = model_name.split("/")
            simple_name = name_parts[-1]
            available_models.append(simple_name)
                    
        logger.info(f"Доступные модели Gemini: {available_models}")
        
        model_priority = [
            "gemini-2.5-flash-preview-05-20", # Актуальная предварительная версия Flash 2.5
            "gemini-2.5-flash-preview-04-17", # Еще одна предварительная версия Flash 2.5
            "gemini-1.5-flash-latest",        # Актуальная версия Flash 1.5
            "gemini-1.5-flash",               # Общая версия Flash 1.5
            "gemini-2.5-pro-preview-05-06",   # Актуальная предварительная версия Pro 2.5 (платная/строгие квоты)
            "gemini-2.5-pro-exp-03-25",       # Еще одна предварительная версия Pro 2.5
            "gemini-1.5-pro-latest",          # Актуальная версия Pro 1.5 (строгие квоты)
            "gemini-1.5-pro",                 # Общая версия Pro 1.5 (строгие квоты)
            "gemini-pro"                      # Оригинальная Pro модель (строгие квоты)
        ]
        
        selected_model = None
        
        for model_name_in_priority in model_priority: # Имя переменной изменено для ясности
            if model_name_in_priority in available_models:
                selected_model = model_name_in_priority
                logger.info(f"Выбрана модель по приоритету: {selected_model}")
                break
                
        if not selected_model and available_models:
            selected_model = available_models[0]
            logger.info(f"Используется первая доступная модель: {selected_model}")
            
        if not selected_model:
            logger.error("Не найдено ни одной доступной модели Gemini!")
            return None
            
        # Инициализируем модель LangChain
        # Передаем сюда selected_model, который теперь соответствует именам из genai.list_models()
        llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=api_key)
        logger.info(f"Языковая модель {selected_model} инициализирована.")
        return llm
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {e}")
        return None

# Определение инструментов (Tools) для агента
@tool
def say_hello(name: str) -> str:
    """Поприветствовать пользователя по имени."""
    return f"Привет, {name}!"

@tool
def calculate(expression: str) -> str:
    """Вычислить математическое выражение.
    
    Args:
        expression: Строка с математическим выражением, например "2 + 2" или "5 * 10"
    
    Returns:
        Результат вычисления как строка
    """
    try:
        # Безопасное выполнение выражения (eval может быть опасен в реальных приложениях)
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат вычисления {expression} = {result}"
    except Exception as e:
        return f"Ошибка при вычислении: {e}"

def create_agent(llm, tools):
    """Создание агента с инструментами."""
    # Создаем более детальный промпт для агента
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Ты полезный AI ассистент, способный помогать пользователям решать различные задачи.
            У тебя есть доступ к специальным инструментам, которые ты можешь использовать.
            Тщательно анализируй запрос пользователя, чтобы определить, какой инструмент лучше использовать.
            Если подходящего инструмента нет, просто отвечай на основе своих знаний."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    try:
        # Создаем агента с поддержкой вызова инструментов
        agent = create_tool_calling_agent(llm, tools, prompt)
        # Создаем исполнителя агента
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    except Exception as e:
        logger.error(f"Ошибка при создании агента: {e}")
        return None

def main():
    """Основная функция программы."""
    # Загрузка API ключа
    api_key = load_api_key()
    if not api_key:
        return
    
    # Инициализация модели
    llm = initialize_llm(api_key)
    if not llm:
        return
    
    # Список всех доступных инструментов
    tools = [say_hello, calculate]
    
    # Создание агента
    agent_executor = create_agent(llm, tools)
    if not agent_executor:
        return
    
    # Основной цикл взаимодействия
    print("\n=== AI Агент на базе Google Gemini ===")
    print("Введите 'выход' для завершения программы.")
    
    while True:
        try:
            user_input = input("\nВаш запрос: ")
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("Завершение работы.")
                break
                
            print(f"\nОбрабатываю запрос: '{user_input}'")
            # Выполняем запрос через AgentExecutor
            response = agent_executor.invoke({"input": user_input})
            print("\nОтвет агента:")
            print(response["output"])
            
        except KeyboardInterrupt:
            print("\nПрограмма прервана пользователем.")
            break
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            print(f"\nПроизошла ошибка: {e}")
            print("Попробуйте сформулировать запрос по-другому.")

if __name__ == "__main__":
    main()