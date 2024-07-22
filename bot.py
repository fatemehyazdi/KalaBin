import logging
from telegram import Update, InputMediaPhoto
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from telegram.error import NetworkError

# توکن بات تلگرام
TOKEN = ''

# مدل تحلیل حساسیت
MODEL_NAME = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# تنظیمات لاگ‌گیری
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('سلام! لینک محصول را ارسال کنید.')

def fetch_product_info(url):
    try:
        # استخراج اطلاعات محصول از سایت
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # استخراج نام محصول
        title = soup.find('h1').text.strip()

        # استخراج عکس محصول
        img_url = soup.find('img')['src']

        # استخراج نظرات
        comments = [comment.text for comment in soup.find_all('div', class_='comment-text')]

        return title, img_url, comments
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException: {e}")
        return None, None, []

def analyze_sentiment(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = probs.detach().numpy()
    positive_scores = [score[1] for score in scores]
    return np.mean(positive_scores)

def product_handler(update: Update, context: CallbackContext) -> None:
    url = update.message.text
    title, img_url, comments = fetch_product_info(url)
    if title and img_url and comments:
        sentiment_score = analyze_sentiment(comments)
        sentiment_percentage = sentiment_score * 100

        update.message.reply_text(f'نام محصول: {title}\n\nدرصد رضایت: {sentiment_percentage:.2f}%')
        update.message.reply_photo(photo=img_url, caption=f'نام محصول: {title}\n\nدرصد رضایت: {sentiment_percentage:.2f}%')
    else:
        update.message.reply_text('خطایی در استخراج اطلاعات محصول رخ داد. لطفاً دوباره تلاش کنید.')

def main() -> None:
    try:
        application = ApplicationBuilder().token(TOKEN).read_timeout(60).connect_timeout(60).build()

        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, product_handler))

        application.run_polling()
    except NetworkError as e:
        logging.error(f"NetworkError: {e}")
        print("لطفاً اتصال اینترنت خود را بررسی کنید یا از VPN استفاده کنید.")

if __name__ == '__main__':
    main()


