import telegram

def sendTelegram(text):
    # This function sends a telegram message

    bot = telegram.Bot(token='1936767568:AAGRCyoOJwcy6rOOZJ1Ol5sxUYZz0vSwL4Q')
    bot.sendMessage(chat_id=803172018,text=text)