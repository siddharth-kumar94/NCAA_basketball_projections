#!/usr/bin/env python3

from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd

column_names = ['player_id', 'year', 'player', 'pos', 'age', 'team_id', 'g', 'gs', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct', 'fg3_per_g', 'fg3a_per_g', 'fg3_pct', 'fg2_per_g',
                'fg2a_per_g', 'fg2_pct', 'efg_pct', 'ft_per_g', 'fta_per_g', 'ft_pct', 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g']


def get_player_data(year):
    season_url = 'https://www.basketball-reference.com/leagues/NBA_{}_per_game.html'.format(
        year)
    season_page = requests.get(season_url).text
    season_soup = BeautifulSoup(season_page, "lxml")
    players = list()
    for tr in season_soup.find('table', id='per_game_stats').tbody.find_all('tr', attrs={'class': 'full_table'}):
        player_id = tr.find(
            'td', attrs={'data-stat': 'player'})['data-append-csv']
        player = dict()
        player.update({'player_id': player_id})
        player.update({'year': year})

        for td in tr.find_all('td'):
            stat_name = td['data-stat']
            stat_val = td.text
            player.update({stat_name: stat_val})
        players.append(player)
    players_df = pd.DataFrame(players, columns=column_names)
    players_df.to_csv('//Users//sidman94//Desktop//NCAA_basketball_project//nba_stats_per_game.csv',
                      encoding='utf-8', mode='a', header=False, index=False)

def execute(year):
    get_player_data(year)
