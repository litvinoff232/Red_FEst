import telebot
from telebot import types
from neuro import MistralChat
from dotenv import load_dotenv
import os
from database import*
load_dotenv()


bot = telebot.TeleBot(str(os.getenv("API")) + ":" + str(os.getenv("API2")))
mistral = MistralChat()





@bot.message_handler(commands=["start"])
def start_message(message):
    try:
        user = message.from_user
        markup = types.InlineKeyboardMarkup(row_width=1)
        btn1 = types.InlineKeyboardButton("Нейросеть", callback_data="neuro")
        btn2 = types.InlineKeyboardButton("График роста☝️", callback_data="grafik")
        markup.add(btn1, btn2)
        bot.send_message(message.chat.id, 
                    '👋Привет! Я бот, который следит за теплицей.\n 👇Выбери чем я могу тебе помочь👇',
                    reply_markup=markup)
        
        
        con = sql_connection()
        cursor = con.cursor()
        sql_table(con)
        
        cursor.execute('SELECT id FROM users WHERE id = ?', (user.id,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # Обновляем данные существующего пользователя
            cursor.execute('''
                UPDATE users 
                SET name = ?,
                id = ?
            ''', (user.username, user.id))
            con.commit()
        
        else:
            cursor.execute('''INSERT INTO users(id, name) 
                        VALUES(?, ?)''', (user.id, user.username))
            con.commit()
            
    finally:
        con.close()
    
    
@bot.callback_query_handler()    
@bot.message_handler(func=lambda msg: True)
def neuro_message(message):
        user_message = message.text
        waiting_message = bot.send_message(message.chat.id, text="Ожидаем...")
        bot.edit_message_text(chat_id= message.chat.id,
                                  message_id=waiting_message.id,
                                  text=mistral.chat(user_message))
        
        
        

if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling(none_stop=True)