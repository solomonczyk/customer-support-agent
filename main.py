import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import tool
# ИСПРАВЛЕННЫЙ ИМПОРТ ДЛЯ SERPER.DEV (правильный класс и путь)
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper 
from datetime import datetime

# Загрузка переменных окружения из .env файла
load_dotenv()

# Определение инструментов
@tool
def say_hello(name: str) -> str:
    """Говорит привет указанному человеку."""
    return f"Привет, {name}!"

@tool
def calculate(expression: str) -> str:
    """Вычисляет математическое выражение.
    Например, '2+2' или '10/3'.
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Ошибка при вычислении: {e}"

@tool
def get_current_datetime() -> str:
    """Возвращает текущую дату и время в читаемом формате."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def serper_search(query: str) -> str:
    """Использует Serper.dev API для поиска информации по заданному запросу.
    Полезен, когда нужно найти актуальную информацию, новости или ответы на вопросы,
    которых нет во внутренней базе знаний.
    """
    print(f"--- ВЫЗВАН serper_search с запросом: '{query}' ---")
    
    # Теперь ищем SERPER_API_KEY, как и положено для GoogleSerperAPIWrapper
    serper_api_key_env = os.getenv("SERPER_API_KEY") 

    if not serper_api_key_env:
        print("--- ОШИБКА: SERPER_API_KEY не настроен. Проверьте .env ---")
        return "Ошибка: SERPER_API_KEY не настроен. Пожалуйста, проверьте файл .env."

    try:
        # Инициализируем GoogleSerperAPIWrapper, он сам возьмет ключ из env
        search = GoogleSerperAPIWrapper() 
        result = search.run(query)
        print(f"--- Serper API вернул результат (часть): {result[:200]}... ---")
        return result
    except Exception as e:
        print(f"--- ОШИБКА Serper API: {e} ---")
        return f"Произошла ошибка при поиске информации: {e}"


def main():
    # Проверка ключа Gemini
    google_api_key = os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        print("Ошибка: Переменная окружения 'GEMINI_API_KEY' не установлена.")
        print("Пожалуйста, установите ее в файле .env.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)

    # Список всех доступных инструментов
    tools = [say_hello, calculate, get_current_datetime, serper_search]

    # --- ВРЕМЕННЫЙ ТЕСТ ДЛЯ serper_search ---
    print("\n--- НАЧИНАЕТСЯ ТЕСТИРОВАНИЕ serper_search НАПРЯМУЮ ---")
    test_query = "Сколько лет Пугачёвой"
    search_result = serper_search.invoke(test_query)
    print(f"--- Результат прямого вызова serper_search: {search_result} ---")
    print("--- ТЕСТИРОВАНИЕ serper_search НАПРЯМУЮ ЗАВЕРШЕНО ---\n")
    # --- КОНЕЦ ВРЕМЕННОГО ТЕСТА ---


    # Создание промпта для агента
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты полезный AI-агент поддержки клиентов. Отвечай на вопросы пользователя."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    chat_history = []

    print("\nВаш запрос: ")
    while True:
        user_input = input()
        if user_input.lower() == "выход":
            print("Завершение работы.")
            break

        print(f"\nОбрабатываю запрос: '{user_input}'\n")

        # Добавляем историю чата и текущий ввод в запрос
        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        print(f"\nОтвет агента:\n{response['output']}\n")
        # Исправляем роль для сообщений агента с 'agent' на 'ai'
        chat_history.extend([{"role": "user", "content": user_input}, {"role": "ai", "content": response["output"]}]) 
        print("\nВаш запрос:")

if __name__ == "__main__":
    main()