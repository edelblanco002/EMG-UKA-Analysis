import telegram

def sendTelegram(text):
    # This function sends a telegram message

    bot = telegram.Bot(token='1936767568:AAGRCyoOJwcy6rOOZJ1Ol5sxUYZz0vSwL4Q')

    if len(text) > 4096:
        for x in range(0, len(text), 4096):
            bot.send_message(chat_id=803172018, text=text[x:x+4096])
    else:
        bot.send_message(chat_id=803172018, text=text)