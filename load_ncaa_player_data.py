#!/usr/bin/env python3
from bs4 import BeautifulSoup
from bs4 import Comment
import requests
import pandas as pd
import csv
import numpy as np

roster_columns = ['player_id', 'player_name', 'year', 'school_id', 'number',
                  'class', 'pos', 'height', 'weight', 'hometown', 'high_school', 'summary']
per_game_columns = ['player_id', 'year', 'g', 'gs', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct', 'fg2_per_g', 'fg2a_per_g', 'fg2_pct', 'fg3_per_g', 'fg3a_per_g',
                    'fg3_pct', 'ft_per_g', 'fta_per_g', 'ft_pct', 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g']
adv_columns = ['player_id', 'year', 'g', 'gs', 'mp', 'per', 'ts_pct', 'efg_pct', 'fg3a_per_fga_pct', 'fta_per_fga_pct', 'pprod', 'orb_pct',
               'drb_pct', 'trb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_per_40', 'obpm', 'dbpm', 'bpm']


def get_roster_data(school_soup, year, school_id):
    players = list()
    for tr in school_soup.find('table', id='roster').tbody.find_all('tr'):
        player = dict()
        player_id = tr.th.a['href'][13:-5]
        player_name = tr.th.a.text
        player.update({'player_id': player_id})
        player.update({'player_name': player_name})
        player.update({'year': year})
        player.update({'school_id': school_id})

        for td in tr.find_all('td'):
            attr = td['data-stat']
            val = td.text
            player.update({attr: val})
        players.append(player)
    players_df = pd.DataFrame(players, columns=roster_columns)
    return players_df


def get_per_game_data(school_soup, year):
    players = list()
    for tr in school_soup.find('table', id='per_game').tbody.find_all('tr'):
        player = dict()
        for td in tr.find_all('td'):
            if td['data-stat'] == 'player':
                player_id = td['data-append-csv']
                player.update({'player_id': player_id})
                player.update({'year': year})
            else:
                if 'iz' not in td['class']:
                    attr = td['data-stat']
                    val = td.text
                    player.update({attr: val})
        players.append(player)
    players_df = pd.DataFrame(players, columns=per_game_columns)
    return players_df


def get_adv_data(school_soup, year):
    players = list()
    for comment in school_soup.find_all(string=lambda text: isinstance(text, Comment)):
        if comment.strip().startswith('<'):
            soup = BeautifulSoup(comment.strip(), 'lxml')
            for table in soup.find_all('table', attrs={'id': 'advanced'}):
                for tr in table.tbody.find_all('tr'):
                    player = dict()
                    for td in tr.find_all('td'):
                        if td['data-stat'] == 'player':
                            player_id = td['data-append-csv']
                            player.update({'player_id': player_id})
                            player.update({'year': year})
                        else:
                            attr = td['data-stat']
                            val = td.text
                            player.update({attr: val})
                    players.append(player)
    players_df = pd.DataFrame(players, columns=adv_columns)
    return players_df


def get_player_data(school_id, year):
    school_url = 'https://www.sports-reference.com/cbb/schools/{}/{}.html'.format(
        school_id, year)
    try:
        school_page = requests.get(school_url).text
        school_soup = BeautifulSoup(school_page, 'lxml')

        #roster_df = get_roster_data(school_soup, year, school_id)
        #per_game_df = get_per_game_data(school_soup, year)
        adv_df = get_adv_data(school_soup, year)

    except Exception as e:
        print(school_id + '-' + str(year) + ': ' + str(e))
        roster_df = pd.DataFrame()
        per_game_df = pd.DataFrame()
    #return roster_df, per_game_df, adv_df
    return adv_df


def get_all_player_data(year):
    season_url = 'https://www.sports-reference.com/cbb/seasons/{}-school-stats.html'.format(
        year)
    season_page = requests.get(season_url).text
    season_soup = BeautifulSoup(season_page, 'lxml')
    all_roster = pd.DataFrame(columns=roster_columns)
    all_per_game = pd.DataFrame(columns=per_game_columns)
    all_adv = pd.DataFrame(columns=adv_columns)

    for tr in season_soup.find('table', id='basic_school_stats').tbody.find_all('tr', attrs={'class': None}):
        school_url = tr.find(
            'td', attrs={'data-stat': 'school_name'}).a['href']
        school_id = school_url[13:-10]
        #roster, per_game, adv = get_player_data(school_id, year)
        adv = get_player_data(school_id, year)
        #all_roster = all_roster.append(roster)
        #all_per_game = all_per_game.append(per_game)
        all_adv = all_adv.append(adv)

    #all_roster.to_csv('//Users//sidman94//Desktop//NCAA_basketball_project//ncaa_players.csv',
    #                  mode='a', header=False, index=False)
    #all_per_game.to_csv('//Users//sidman94//Desktop//NCAA_basketball_project//ncaa_per_game.csv',
    #                    mode='a', header=False, index=False)
    all_adv.to_csv('//Users//sidman94//Desktop//NCAA_basketball_project//ncaa_adv.csv',
                   columns=adv_columns, mode='a', header=False, index=False)


def execute(year):
    get_all_player_data(year)
