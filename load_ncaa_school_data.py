#!/usr/bin/env python3

from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv

columns = ['year', 'school_id', 'g', 'wins', 'losses', 'win_loss_pct', 'srs', 'sos', 'wins_conf', 'losses_conf', 'wins_home', 'losses_home', 'wins_visitor', 'losses_visitor', 'pts', 'opp_pts', 'mp', 'fg', 'fga', 'fg_pct', 'fg3', 'fg3a', 'fg3_pct', 'ft', 'fta', 'ft_pct', 'orb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']

def get_school_season_data(year):
    season_url = "https://www.sports-reference.com/cbb/seasons/{}-school-stats.html".format(
        year)
    season_page = requests.get(season_url).text
    season_soup = BeautifulSoup(season_page, "lxml")

    schools = list()
    for tr in season_soup.find(name='table', id='basic_school_stats').tbody.find_all(name='tr', attrs={'class': None}):
        school = dict()
        school.update({'year': year})
        for td in tr.find_all('td'):
            if td['data-stat'] != 'x':
                if td['data-stat'] == 'school_name':
                    school_url = td.a['href']
                    school_id = school_url[13:-10]
                    school.update({'school_id': school_id})
                else:
                    stat_name = td['data-stat']
                    stat_val = td.text
                    school.update({stat_name: stat_val})
        schools.append(school)
    schools_df = pd.DataFrame(schools, columns=columns)
    schools_df.to_csv(
        '//Users//sidman94//Desktop//NCAA_basketball_project//ncaa_school_stats.csv', mode='a', header=False, index=False)

def execute(year):
    get_school_season_data(year)
