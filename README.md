# MovieLens Recommender

Двухэтапная рекомендательная система на MovieLens 1M:
**Implicit ALS + ItemKNN → CatBoostRanker → Streamlit**.

Проект сравнивает MostPopular, ALS, ItemKNN и CatBoost с временным разделением
данных и полным ground truth. На февральском test NDCG@10 составляет **0.10233**
у CatBoost против **0.10008** у ItemKNN; преимущество небольшое.

- [Описание проекта, результаты и запуск](movielens-reccomender/README.md)
- [Ноутбуки экспериментов](notebooks/)
- [Код моделей и оценки](movielens-reccomender/src/movielens_recsys/)
- [Streamlit-приложение](movielens-reccomender/app/README.md)

## Воспроизведение

1. Скачайте [MovieLens 1M у GroupLens](https://grouplens.org/datasets/movielens/1m/).
2. Распакуйте архив и поместите `ratings.dat`, `movies.dat`, `users.dat` в корень
   этого репозитория, рядом с этим файлом.
3. Установите зависимости и выполните ноутбуки по
   [инструкции запуска](movielens-reccomender/README.md#данные-и-запуск).

Датасет, обученные модели и локальное Python-окружение исключены из Git.
Результаты экспериментов доступны в выводах ноутбуков и таблицах метрик.
Описание формата исходных данных предоставлено GroupLens в файле [README](README).
