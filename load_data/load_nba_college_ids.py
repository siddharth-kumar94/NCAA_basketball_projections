#!/usr/bin/env python3

from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import re


def get_college_player_id(player_id):
    player_url = 'https://www.basketball-reference.com/players/{}/{}.html'.format(
        player_id[0], player_id)
    player_page = requests.get(player_url).text
    player_soup = BeautifulSoup(player_page, 'lxml')
    college_id = ''
    for li in player_soup.find_all('li'):
        a = li.find('a', attrs={'href': re.compile('cbb/players')})
        if a is not None:
            college_id = a['href'][45:-5]
    return college_id


def get_player_data(year):
    existing_players = pd.read_csv(
        '//Users//sidman94//Desktop//NCAA_basketball_project//nba_players.csv')
    season_url = 'https://www.basketball-reference.com/leagues/NBA_{}_per_game.html'.format(
        year)
    season_page = requests.get(season_url).text
    season_soup = BeautifulSoup(season_page, "lxml")
    new_players = list()
    for tr in season_soup.find('table', id='per_game_stats').tbody.find_all('tr', attrs={'class': 'full_table'}):
        player_id = tr.find(
            'td', attrs={'data-stat': 'player'})['data-append-csv']
        if ~existing_players['player_id'].str.contains(player_id).any():
            college_player_id = get_college_player_id(player_id)
            if len(college_player_id) > 0:
                player = dict()
                player.update({'player_id': player_id})
                player.update({'college_player_id': college_player_id})
                new_players.append(player)
    new_players_df = pd.DataFrame(columns=['player_id', 'college_player_id'])
    new_players_df.append(pd.DataFrame(new_players))
    new_players_df.set_index('player_id', inplace=True)
    new_players_df.to_csv(
        '//Users//sidman94//Desktop//NCAA_basketball_project//nba_players.csv', mode='a', header=False)

def execute(year):
    get_player_data(year)
