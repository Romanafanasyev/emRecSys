from flask import Flask, render_template, request, redirect, url_for
import os
import random
import pickle
from time import sleep
from deepface import DeepFace
import cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Директория с изображениями
IMAGE_DIR = './static/'

# Директория для сохранения фотографий с вебкамеры
FACEPIC_DIR = './facepic/'

# Количество изображений для рекомендации
IMAGES_TO_RECOMMEND_NUM = 5

# Путь к базе данных
DB_PATH = './database.pkl'

# Загружаем базу данных или создаем новую, если ее нет
if os.path.exists(DB_PATH):
    with open(DB_PATH, 'rb') as db_file:
        users_db = pickle.load(db_file)
else:
    users_db = {}


# Функция для генерации списка случайных изображений
def generate_random_rec():
    all_images = os.listdir(IMAGE_DIR)
    return random.sample(all_images, IMAGES_TO_RECOMMEND_NUM)


# Коллаборативный алгоритм основанный на сходстве между пользователями
def recommend_images(login):
    with open(DB_PATH, 'rb') as file:
        data = pickle.load(file)

    df = pd.DataFrame(data).T
    df['emotions'].fillna(False, inplace=True)
    # Фильтрация пользователей без эмоций
    df = df[df['emotions'].apply(lambda x: bool(x))]
    # Создание матрицы сходства на основе эмоций
    emotions_matrix = pd.get_dummies(df['emotions'].apply(pd.Series).stack()).groupby(level=0).sum()
    similarity_matrix = cosine_similarity(emotions_matrix)
    # Находим индекс пользователя
    user_index = df.index.get_loc(login)
    # Вычисляем схожих пользователей
    similar_users = pd.Series(similarity_matrix[user_index], index=df.index)
    # Сортируем пользователей по убыванию схожести и убираем самого пользователя
    similar_users = similar_users.sort_values(ascending=False)[1:]
    # Получаем рекомендации изображений от схожих пользователей
    recommended_images = []
    for user in similar_users.index:
        # Фильтруем эмоции пользователя, оставляя только положительные
        positive_emotions = df.loc[user, 'emotions']
        positive_emotions = {img: emo for img, emo in positive_emotions.items() if emo in ['happy', 'surprise']}
        # Добавляем рекомендации изображений пользователя с положительными эмоциями
        recommended_images.extend(df.loc[user, 'images_to_recommend'])
    # Убираем дубликаты из рекомендаций
    recommended_images = list(set(recommended_images))
    # Если не набрали достаточно
    recommended_images += generate_random_rec()
    return recommended_images


@app.route('/', methods=['GET', 'POST'])
def login():
    global users_db

    if request.method == 'POST':
        login = request.form['login']

        # Создаем пользователя, если его еще нет в базе данных
        if login not in users_db:
            users_db[login] = {'images_to_recommend': generate_random_rec(), 'image_counter': 0}

        # Сохраняем информацию о пользователе в базе данных
        with open(DB_PATH, 'wb') as db_file:
            pickle.dump(users_db, db_file)

        # Перенаправляем пользователя на страницу с изображениями
        return redirect(url_for('images', login=login))

    return render_template('login.html')


@app.route('/images/<login>', methods=['GET', 'POST'])
def images(login):
    global users_db

    user_data = users_db.get(login)

    if request.method == 'POST':
        # При нажатии кнопки Вперед
        if request.form['action'] == 'next':
            user_data['image_counter'] += 1
            image_counter = user_data['image_counter']

            if image_counter < IMAGES_TO_RECOMMEND_NUM:
                current_image = user_data['images_to_recommend'][image_counter]

                # Ждем 1 секунду
                sleep(1)

                # Делаем фотографию с вебкамеры и сохраняем в ./facepic
                webcam = cv2.VideoCapture(0)
                _, frame = webcam.read()
                webcam.release()

                facepic_path = os.path.join(FACEPIC_DIR, f"{login}_image_{image_counter}.jpg")
                cv2.imwrite(facepic_path, frame)

                # Определяем эмоцию с помощью deepface
                try:
                    emotions = DeepFace.analyze(facepic_path, actions=['emotion'])
                    print(emotions)
                    user_emotion = emotions[0]['dominant_emotion']
                except Exception as e:
                    print(f"Error analyzing emotion: {e}")
                    user_emotion = 'unknown'

                # Сохраняем запись об эмоции пользователя в базе данных
                user_data['emotions'] = user_data.get('emotions', {})
                user_data['emotions'][current_image] = user_emotion

                # Удаляем изображение из ./facepic
                os.remove(facepic_path)

                # Сохраняем информацию о пользователе в базе данных
                with open(DB_PATH, 'wb') as db_file:
                    pickle.dump(users_db, db_file)

                print(f"Определена эмоция: {user_emotion}")

            else:

                # Если счетчик изображений достиг IMAGES_TO_RECOMMEND_NUM, обновляем рекомендации
                user_data['image_counter'] = 0
                user_data['images_to_recommend'] = recommend_images(login)

                print("new recs:", user_data['images_to_recommend'])

                # Сохраняем информацию о пользователе в базе данных
                with open(DB_PATH, 'wb') as db_file:
                    pickle.dump(users_db, db_file)

                current_image = user_data['images_to_recommend'][user_data['image_counter']]
                print("Обновили рекомендации")

    else:
        # Получаем текущее изображение
        current_image = user_data['images_to_recommend'][user_data['image_counter']]

    return render_template('images.html', image=current_image, login=login)


# Сохраняем базу данных при завершении приложения
@app.teardown_appcontext
def save_db_on_exit(exception=None):
    with open(DB_PATH, 'wb') as db_file:
        pickle.dump(users_db, db_file)

if __name__ == '__main__':
    app.run(debug=True)
