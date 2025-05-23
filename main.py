import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import tool
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper 
from datetime import datetime
import json 
import uuid 
from agent.graph_builder import build_graph
from dotenv import load_dotenv
load_dotenv()        # подхватывает файл .env рядом с проектом

if __name__ == "__main__":
    graph = build_graph()
    while True:
        user = input("⮕  ")
        if user.lower() in {"exit", "quit"}:
            break
        res = graph.invoke({"user_input": user})
        print(res["response"])


# Загрузка переменных окружения из .env файла
load_dotenv()

# --- ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ ---

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
    
    serper_api_key_env = os.getenv("SERPER_API_KEY") 

    if not serper_api_key_env:
        print("--- ОШИБКА: SERPER_API_KEY не настроен. Проверьте .env ---")
        return "Ошибка: SERPER_API_KEY не настроен. Пожалуйста, проверьте файл .env."

    try:
        search = GoogleSerperAPIWrapper() 
        result = search.run(query)
        print(f"--- Serper API вернул результат (часть): {result[:200]}... ---")
        return result
    except Exception as e:
        print(f"--- ОШИБКА Serper API: {e} ---")
        return f"Произошла ошибка при поиске информации: {e}"

@tool
def get_from_knowledge_base(query: str) -> str:
    """Использует внутреннюю базу знаний для поиска ответов на часто задаваемые вопросы.
    Полезен для получения информации о доставке, возвратах, характеристиках продуктов и т.д.
    Используй этот инструмент, если вопрос пользователя похож на запрос к FAQ или внутренней информации.
    """
    print(f"--- ВЫЗВАН get_from_knowledge_base с запросом: '{query}' ---")
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            kb_data = json.load(f)
        
        query_lower = query.lower().strip() 
        
        found_answers = []
        for item in kb_data:
            if item.get("question", "").lower().strip() == query_lower:
                found_answers.append(f"Категория: {item['category']}\nВопрос: {item['question']}\nОтвет: {item['answer']}")
                break 
        
        if found_answers:
            print(f"--- Найдено ответов в базе знаний: {len(found_answers)} ---")
            return "\n\n".join(found_answers)
        else:
            for item in kb_data:
                if query_lower in item.get("question", "").lower() or \
                   query_lower in item.get("category", "").lower():
                    found_answers.append(f"Категория: {item['category']}\nВопрос: {item['question']}\nОтвет: {item['answer']}")
            
            if found_answers:
                print(f"--- Найдено ответов в базе знаний (по подстроке): {len(found_answers)} ---")
                return "\n\n".join(found_answers)
            else:
                print("--- В базе знаний не найдено подходящих ответов. ---")
                return "Внутренняя база знаний не содержит информации по вашему запросу."
    except FileNotFoundError:
        print("--- ОШИБКА: Файл knowledge_base.json не найден. ---")
        return "Ошибка: Внутренняя база знаний недоступна."
    except json.JSONDecodeError:
        print("--- ОШИБКА: Ошибка чтения JSON файла knowledge_base.json. ---")
        return "Ошибка: Внутренняя база знаний повреждена."
    except Exception as e:
        print(f"--- ОШИБКА в get_from_knowledge_base: {e} ---")
        return f"Произошла ошибка при доступе к внутренней базе знаний: {e}"

@tool
def store_user_preference(preference_key: str, preference_value: str) -> str:
    """Сохраняет предпочтение пользователя для персонализации будущих ответов.
    Используй этот инструмент, когда пользователь явно указывает свои предпочтения,
    например, предпочитаемый способ связи, имя, регион или другие детали.
    preference_key: ключ для предпочтения (например, 'имя', 'способ_связи', 'регион').
    preference_value: значение предпочтения (например, 'Андрей', 'email', 'Москва').
    """
    print(f"--- ВЫЗВАН store_user_preference: ключ='{preference_key}', значение='{preference_value}' ---")
    return f"Предпочтение '{preference_key}' со значением '{preference_value}' сохранено для будущих ответов."

@tool
def create_support_ticket(issue_description: str, user_email: str = None) -> str:
    """Создает новый тикет в системе поддержки с описанием проблемы пользователя.
    Используй этот инструмент, когда пользователь явно просит о помощи с проблемой, которую агент не может решить напрямую,
    или когда требуется дальнейшее рассмотрение специалистом.
    issue_description: подробное описание проблемы, с которой столкнулся пользователь.
    user_email: (необязательно) адрес электронной почты пользователя для отправки подтверждения.
    """
    ticket_id = str(uuid.uuid4())[:8] 
    
    print(f"--- ВЫЗВАН create_support_ticket: описание='{issue_description}', email='{user_email}' ---")
    
    response_message = (
        f"Тикет поддержки №{ticket_id} успешно создан.\n"
        f"Описание проблемы: '{issue_description}'.\n"
    )
    if user_email:
        response_message += f"Подтверждение будет отправлено на ваш email: {user_email}."
    else:
        response_message += "Наш специалист свяжется с вами в ближайшее время."
        
    return response_message

@tool
def perform_website_action(action_type: str, details: str) -> str:
    """Имитирует выполнение действия на веб-сайте, такого как сброс пароля,
    проверка статуса заказа, обновление профиля и т.д.
    Используй этот инструмент, когда пользователь просит выполнить конкретное действие на сайте,
    но ты сам не можешь его выполнить напрямую.
    action_type: Тип действия, которое нужно выполнить (например, 'сброс пароля', 'проверить статус заказа', 'обновить адрес').
    details: Дополнительные детали, необходимые для выполнения действия (например, email, номер заказа, новый адрес).
    """
    print(f"--- ВЫЗВАН perform_website_action: тип='{action_type}', детали='{details}' ---")
    
    if action_type.lower() == "сброс пароля":
        return f"Действие '{action_type}' для {details} имитировано: Инструкции по сбросу пароля отправлены на {details}."
    elif action_type.lower() == "проверить статус заказа":
        return f"Действие '{action_type}' для заказа {details} имитировано: Статус заказа {details} - 'В обработке', ожидаемая дата доставки 2-3 дня."
    elif action_type.lower() == "обновить адрес":
        return f"Действие '{action_type}' для адреса '{details}' имитировано: Ваш адрес успешно обновлен до '{details}'."
    else:
        return f"Действие '{action_type}' с деталями '{details}' имитировано: Запрос на выполнение действия получен. Специалист скоро свяжется с вами."

# --- ЛОГИКА УПРАВЛЕНИЯ СЕССИЯМИ ---

SESSION_DIR = "sessions"

def load_session_history(session_id: str) -> list:
    """Загружает историю чата для данной сессии из файла."""
    session_file_path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(session_file_path):
        with open(session_file_path, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
                print(f"--- История сессии '{session_id}' загружена. ---")
                return history
            except json.JSONDecodeError:
                print(f"--- Ошибка чтения истории сессии '{session_id}'. Создаем новую. ---")
                return []
    print(f"--- Новая сессия '{session_id}' инициализирована. ---")
    return []

def save_session_history(session_id: str, chat_history: list):
    """Сохраняет историю чата для данной сессии в файл."""
    os.makedirs(SESSION_DIR, exist_ok=True) 
    session_file_path = os.path.join(SESSION_DIR, f"{session_id}.json")
    with open(session_file_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)
    print(f"--- История сессии '{session_id}' сохранена. ---")


def main():
    # Проверка ключа Gemini
    google_api_key = os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        print("Ошибка: Переменная окружения 'GEMINI_API_KEY' не установлена.")
        print("Пожалуйста, установите ее в файле .env.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)

    # Список всех доступных инструментов
    tools = [
        say_hello,
        calculate,
        get_current_datetime,
        serper_search,
        get_from_knowledge_base,
        store_user_preference,
        create_support_ticket,
        perform_website_action 
    ]

    # Создание промпта для агента
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты полезный AI-агент поддержки клиентов. Всегда стремись быть дружелюбным и полезным. "
                       "Используй предоставленные инструменты для поиска информации и выполнения действий. "
                       "Если пользователь указывает свои предпочтения (например, имя, способ связи, город), "
                       "используй инструмент `store_user_preference` для их сохранения и старайся использовать их в будущих ответах для персонализации. "
                       "Всегда проверяй историю чата на предмет ранее сохраненных предпочтений. "
                       "Если пользователь просит о помощи с проблемой, которую ты не можешь решить напрямую, или требует дальнейшего рассмотрения, предложи создать тикет поддержки, используя инструмент `create_support_ticket`. "
                       "Если пользователь просит выполнить действие на сайте (например, сбросить пароль, проверить статус заказа, обновить данные), используй инструмент `perform_website_action`."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # --- ЛОГИКА УПРАВЛЕНИЯ СЕССИЯМИ ---
    session_id = input("Введите ID сессии (или нажмите Enter для новой сессии): ").strip()
    if not session_id:
        session_id = str(uuid.uuid4())[:8] 
        print(f"Начата новая сессия с ID: {session_id}")
    
    chat_history = load_session_history(session_id)
    # --- КОНЕЦ ЛОГИКИ УПРАВЛЕНИЯ СЕССИЯМИ ---

    print("\nВаш запрос: ")
    while True:
        try: 
            user_input = input()
            if user_input.lower() == "выход":
                print("Завершение работы.")
                save_session_history(session_id, chat_history) 
                break

            print(f"\nОбрабатываю запрос: '{user_input}'\n")

            response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            print(f"\nОтвет агента:\n{response['output']}\n")
            
            chat_history.extend([{"role": "user", "content": user_input}, {"role": "ai", "content": response["output"]}]) 
            save_session_history(session_id, chat_history) 
            
            print("\nВаш запрос:")
        except KeyboardInterrupt:
            print("\nПрограмма прервана пользователем. Сохраняю историю...")
            save_session_history(session_id, chat_history) 
            break
        except Exception as e:
            print(f"Произошла непредвиденная ошибка: {e}")
            print("Пожалуйста, попробуйте еще раз.")
            pass 

if __name__ == "__main__":
    main()