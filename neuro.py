from mistralai import Mistral
from dotenv import load_dotenv
import os

# Загружаем переменные окружения
load_dotenv()

class MistralChat:
    def __init__(self):
        # Получаем API-ключ из переменных окружения
        api_key = os.getenv("MISTRAL_API_KEY")
        
        # Проверяем, что ключ существует
        if not api_key:
            raise ValueError("MISTRAL_API_KEY не найден в переменных окружения. Проверьте файл .env")
        
        self.client = Mistral(api_key=api_key)
        self.conversation_history = [
            {
                "role": "system", 
                "content": """Ты опытный агроном, который может ответить на все вопросы. На вопрос отвечать кратко и по делу,
                не более 4000 символов. Ответ должен быть написан обычным шрифтом(без **Агротехнические** и подобного)"""
            }
        ]
    
    def chat(self, message):
        self.conversation_history.append({"role": "user", "content": message})
        
        response = self.client.chat.complete(
            model="mistral-medium",
            messages=self.conversation_history
        )
        
        assistant_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant",
                                          "content": assistant_response})
        
        return assistant_response

# Использование
if __name__ == "__main__":
    try:
        bot = MistralChat()
        response = bot.chat("Привет! Как тебя зовут?")
        print(response)
    except ValueError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")