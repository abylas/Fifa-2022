# testing commits
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
import pandas as pd
from bs4 import BeautifulSoup
import requests
import pickle
from string import ascii_uppercase as alphabet
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson
from pathlib import Path

sns.set_style('darkgrid')
path = "C:/Users/sendt/AppData/Local/Programs/Python/Python310/Scripts"  # Write your path here
# service = Service(executable_path=path)
driver = webdriver.Chrome()


# service=service)


def get_misssing_data(year):
    web = f'https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup'

    driver.get(web)
    matches = driver.find_elements(by='xpath', value='//td[@align="right"]/.. | //td[@style="text-align:right;"]/..')
    # matches = driver.find_elements(by='xpath', value='//tr[@style="font-size:90%"]')

    home = []
    score = []
    away = []

    for match in matches:
        home.append(match.find_element(by='xpath', value='./td[1]').text)
        score.append(match.find_element(by='xpath', value='./td[2]').text)
        away.append(match.find_element(by='xpath', value='./td[3]').text)

    dict_football = {'home': home, 'score': score, 'away': away}
    df_football = pd.DataFrame(dict_football)
    df_football['year'] = year
    time.sleep(2)
    return df_football


years = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974,
         1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014,
         2018]

fifa = [get_misssing_data(year) for year in years]
driver.quit()
df_fifa = pd.concat(fifa, ignore_index=True)
df_fifa.to_csv("fifa_worldcup_missing_data.csv", index=False)

# 1.get_results_and_fixture.py
# data to scraper -> 1930-2018
years = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974,
         1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014,
         2018]


def get_matches(year):
    web = f'https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup'
    response = requests.get(web)
    content = response.text
    soup = BeautifulSoup(content, 'lxml')
    matches = soup.find_all('div', class_='footballbox')

    home = []
    score = []
    away = []

    for match in matches:
        home.append(match.find('th', class_='fhome').get_text())
        score.append(match.find('th', class_='fscore').get_text())
        away.append(match.find('th', class_='faway').get_text())

    dict_football = {'home': home, 'score': score, 'away': away}
    df_football = pd.DataFrame(dict_football)
    df_football['year'] = year
    return df_football


# results: historical data
fifa = [get_matches(year) for year in years]
df_fifa = pd.concat(fifa, ignore_index=True)
df_fifa.to_csv("fifa_worldcup_historical_data.csv", index=False)

# fixture
df_fixture = get_matches(2022)
df_fixture.to_csv('fifa_worldcup_fixture.csv', index=False)


#  3 - 1.get_tables_groupstage.ipynb

# extracting all tables in website
all_tables = pd.read_html('https://en.wikipedia.org/wiki/2022_FIFA_World_Cup')
# finding tables in group stage
all_tables[11]
all_tables[18]
all_tables[25]

# A -> H
# 11 -> 7*8 + 11 = 67
all_tables = pd.read_html('https://en.wikipedia.org/wiki/2022_FIFA_World_Cup')
for i in range(11,67,7): # 11 18 25
    print(i)
    df = all_tables[i]
    df.rename(columns={df.columns[1]:'Team'}, inplace=True)
    df.pop('Qualification')
#     print(df)


print(alphabet)

all_tables = pd.read_html('https://en.wikipedia.org/wiki/2022_FIFA_World_Cup')

dict_table = {}
for letter, i in zip(alphabet, range(11,67,7)): # A=11, B=18, ...
    df = all_tables[i]
    df.rename(columns={df.columns[1]:'Team'}, inplace=True)
    df.pop('Qualification')
    dict_table[f'Group {letter}'] = df
dict_table.keys()

dict_table['Group H']

# Upload (..verify if uploaded correctly)
with open('dict_table', 'wb') as output:
    pickle.dump(dict_table, output)