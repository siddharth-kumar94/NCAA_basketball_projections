#!/usr/bin/env python3

from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd

column_names = ['player_id', 'year', 'player', 'pos', 'age', 'team_id', 'g', 'mp', 'per', 'ts_pct', 'fg3a_per_fga_pct', 'fta_per_fga_pct', 'orb_pct',
                'drb_pct', 'trb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_per_48', 'obpm', 'dbpm', 'bpm', 'vorp']


def get_player_data(year):
    season_url = 'https://www.basketball-reference.com/leagues/NBA_{}_advanced.html'.format(
        year)
    season_page = requests.get(season_url).text
    season_soup = BeautifulSoup(season_page, "lxml")
    new_players = list()
    for tr in season_soup.find('table', id='advanced_stats').tbody.find_all('tr', attrs={'class': 'full_table'}):
        player_id = tr.find(
            'td', attrs={'data-stat': 'player'})['data-append-csv']

        player = dict()
        player.update({'player_id': player_id})
        player.update({'year': year})

        for td in tr.find_all('td'):
            if 'iz' not in td['class']:
                stat_name = td['data-stat']
                stat_val = td.text
                player.update({stat_name: stat_val})
        new_players.append(player)
    new_players_df = pd.DataFrame(new_players, columns=column_names)
    new_players_df.to_csv('//Users//sidman94//Desktop//NCAA_basketball_project//nba_stats_advanced.csv',
                          mode='a', header=False, encoding='utf-8', index=False)

def execute(year):
    get_player_data(year)
