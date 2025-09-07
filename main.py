import telebot
from telebot import types

API = "8364591627:AAHQtqktdglI4iYalpuXhQx41cMvNrcxCws"

bot = telebot.TeleBot(API)

@bot.message_handler(commands=["start"])
def start_message(message):
    markup = types.InlineKeyboardMarkup(row_width=1)
    btn1 = types.InlineKeyboardButton("Отправить отчет🤓", callback_data="otch")
    btn2 = types.InlineKeyboardButton("График роста☝️", callback_data="grafik")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, 
                   '👋Привет! Я бот, который следит за теплицей.\n 👇Выбери чем я могу тебе помочь👇',
                   reply_markup=markup)
    

bot.polling(non_stop=True)
