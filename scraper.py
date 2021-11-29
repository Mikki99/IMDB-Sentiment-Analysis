import csv

from bs4 import BeautifulSoup
import requests
import re
from scrapy.http import TextResponse
from urllib.parse import urljoin
import random

movie_codes = []
with open("data.tsv", "rt", encoding="utf8") as f:
    for line in f:
        line = line.split("\t")
        if line[3] == "US":
            movie_codes.append(line[0])


def random_list(list, num_elements, seed=5):
    random.seed(seed)
    return random.choices(list, k=num_elements)


reviews, ratings = [], []
for code in random_list(movie_codes, 5000, 3):
    url = "https://www.imdb.com/title/{}/reviews/?ref_=tt_ql_urv".format(code)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # print(code)
    data = soup.findAll('div', class_='lister-item-content')
    if data:
        for item in data:
            review = item.find('div', class_='text show-more__control')
            rating = item.find('span', class_='rating-other-user-rating')

            if review and rating:
                reviews.append(review.text)
                rating_label = rating.find('span')
                ratings.append(rating_label.text)

                # print(review.text)
                # print("-------------------------------")
                # print(rating.text)
            # print("================================")
            # print("================================")


print(ratings[0])
print(reviews[0])

with open("movie_reviews2.tsv", "wt", encoding="utf8") as out_f:
    tsv_writer = csv.writer(out_f, delimiter="\t")
    for rating, review in zip(ratings, reviews):
        tsv_writer.writerow([rating, review])


# url = 'http://www.imdb.com/chart/top'
# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'lxml')
#
# movies = soup.select('td.titleColumn')
# # links = [a.attrs.get('href') for a in soup.select('td.titleColumn a')]
# # crew = [a.attrs.get('title') for a in soup.select('td.titleColumn a')]
# ratings = [b.attrs.get('data-value') for b in soup.select('td.posterColumn span[name=ir]')]
# votes = [b.attrs.get('data-value') for b in soup.select('td.ratingColumn strong')]

