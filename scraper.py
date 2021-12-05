import csv
from bs4 import BeautifulSoup
import requests
import random

# Collect English (US) movies from the database
movie_codes = []
with open("data.tsv", "rt", encoding="utf8") as f:
    for line in f:
        line = line.split("\t")
        if line[3] == "US":
            movie_codes.append(line[0])
#print(len(movie_codes))

def random_list(list, num_elements, seed=5):
    """
    Function to pick random elements from a list
    :param list: input list
    :param num_elements: number of elements to pick randomly
    :param seed: integer; used for reproducible results
    :return: random subset of input list with num_elements elements
    """
    random.seed(seed)
    return random.choices(list, k=num_elements)


# Scraping IMDB movie reviews and ratings
# for random choice of 5000 movies from our list
reviews, ratings = [], []
for code in random_list(movie_codes, 10000, 3):
    url = "https://www.imdb.com/title/{}/reviews/?ref_=tt_ql_urv".format(code)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = soup.findAll('div', class_='lister-item-content')
    if data:
        for item in data:
            review = item.find('div', class_='text show-more__control')
            rating = item.find('span', class_='rating-other-user-rating')

            if review and rating:
                reviews.append(review.text)
                rating_label = rating.find('span')
                ratings.append(rating_label.text)

print(ratings[0])
print(reviews[0])

# Save the reviews and ratings into a .tsv file
with open("movie_reviews3.tsv", "wt", encoding="utf8") as out_f:
    tsv_writer = csv.writer(out_f, delimiter="\t")
    for rating, review in zip(ratings, reviews):
        tsv_writer.writerow([rating, review])

