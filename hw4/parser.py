
# Парсер данных с world-art.ru


import json
from bs4 import BeautifulSoup
import requests
import csv


url = 'http://www.world-art.ru/animation/animation.php'
params = {}

with open('data/anime_data.csv', 'w', newline='') as dataset:
    animewriter = csv.writer(dataset, delimiter='\t')
    
    for i in range(2500,2600): #качаем небольшими кусками, и всё равно ловим бан
        params['id'] = i
        r = requests.get(url, params=params)
        soup = BeautifulSoup(r.text, 'lxml')

        plot_found = soup.find("table", string="Краткое содержание") # пытаемся найти сюжет

        if plot_found: #сюжет есть не везде, далеко не везде
            plot_text = plot_found.find_next("table", border="0", cellpadding="2", cellspacing="0", width="100%").text
            genre = soup.find('a', class_="review", href="http://www.world-art.ru/animation/list.php").text
            animewriter.writerow([plot_text, genre])
            print('*', end='') # получилось -- рисуем звёздочку
        else:
            print('x', end='') # в противном случае -- крестик



