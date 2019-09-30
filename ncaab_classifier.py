# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import BayesianRidge, LogisticRegression, LinearRegression, SGDRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR, SVC
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

features_list = ['g', 'gs', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct', 'fg2_per_g', 'fg2a_per_g', 'fg2_pct', 'fg3_per_g', 'fg3a_per_g', 'fg3_pct', 'ft_per_g', 'fta_per_g', 'ft_pct', 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'tenure', 'height', 'weight', 'sos', 'srs', 'ows', 'dws', 'ws', 'ts_pct', 'usg_pct', 'bpm', 'pprod']

def convert_height_to_inches(df):
    inches=0
    if pd.notnull(df['height']):
        height_split = df.loc['height'].split('-')
        inches = int(height_split[0])*12 + int(height_split[1])
    return inches

def scale(df):
    return (sos_os(df['sos']) + 1) * (1/(df['tenure'])) * df

def transform_train_data(X):
    X['per'] = (1/X['mp_per_g']) * ((X['fg_per_g'] * 85.91) + (X['stl_per_g'] * 53.897) + (X['fg3_per_g'] * 51.757) + (X['ft_per_g'] * 46.845) + (X['ast_per_g'] * 34.677) - (X['pf_per_g'] * 17.174) - (X['fta_per_g'] - (X['ft_per_g'])*20.091) - ((X['fga_per_g'] - X['fg_per_g'])*39.19) - (X['tov_per_g']*53.897))
    X = X[X['height'] > 0]
    X['bmi'] = 703 * X['weight'] / (X['height'] **2)
    X['fg3ar'] = X['fg3a_per_g'] / X['fga_per_g']
    X.fillna(0, inplace=True)

    return X

def assign_tier(X):
    per = X['mean_per']
    pos = X['pos']
    if (pos == 'G') | (pos == 'F'):
        if per < 1000:
            return 9
        elif per < 5000:
            return 8
        elif per < 10000:
            return 7
        elif per < 15000:
            return 6
        elif per < 20000:
            return 5
        elif per < 30000:
            return 4
        elif per < 40000:
            return 3
        elif per < 50000:
            return 2
        elif per < 100000:
            return 1
    elif pos == 'C':
        if per < 1000:
            return 7
        elif per < 5000:
            return 6
        elif per < 12000:
            return 5
        elif per < 18000:
            return 4
        elif per < 25000:
            return 3
        elif per < 35000:
            return 2
        elif per < 100000:
            return 1


def predict_make_nba(year, X):

    X_year = X[df['year'] == year]

    #remove data for year being predicted
    X = X[(df['year'] != year) & (df['year'] < 2019)]
    y = df[['made_nba']].loc[X.index]

    clf = LogisticRegression(solver='newton-cg', class_weight='balanced')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    #train classifier
    clf.fit(X_train, y_train)

    #predict if players in given year will make NBA
    pred_to_nba = clf.predict(X_year)
    pred_to_nba = pd.DataFrame(pred_to_nba, index=X_year.index)
    pred_to_nba = pred_to_nba[pred_to_nba[0]]
    pred_to_nba = pred_to_nba.merge(df, left_index=True, right_index=True)
    pred_to_nba.fillna(0, inplace=True)

    zero_sample = pred_to_nba[(pred_to_nba['mean_per'] == 0) & (pred_to_nba['year'] < 2019)].sample(frac=.2)
    pred_to_nba = pred_to_nba[(pred_to_nba['mean_per'] > 0) | (pred_to_nba['year'] == 2019)]
    pred_to_nba = pred_to_nba.append(zero_sample)

    return pred_to_nba

#Read in NCAA data
ncaa_per_game = pd.read_csv('//Users//sidman94//Desktop//NCAA_basketball_project//dat//ncaa_per_game.csv')
ncaa_adv = pd.read_csv('//Users//sidman94//Desktop//NCAA_basketball_project//dat//ncaa_adv.csv')
ncaa_players = pd.read_csv('//Users//sidman94//Desktop//NCAA_basketball_project//dat//ncaa_players.csv')
ncaa_team_stats = pd.read_csv('//Users//sidman94//Desktop//NCAA_basketball_project//dat//ncaa_school_stats.csv')
school_names = pd.read_csv('//Users//sidman94//Desktop//NCAA_basketball_project//dat//school_names.csv')

#Read in NBA data
nba_players = pd.read_csv('//Users//sidman94//Desktop//NCAA_basketball_project//dat//nba_players.csv')
nba_players.rename(columns={'player_id':'nba_player_id'}, inplace=True)

nba_per_game = pd.read_csv('//Users//sidman94//Desktop//NCAA_basketball_project//dat//nba_stats_per_game.csv')
nba_adv_stats = pd.read_csv('//Users//sidman94//Desktop//NCAA_basketball_project//dat//nba_stats_advanced.csv')

#Manipulate NCAA data

#add min and max years
ncaa_per_game['min_year'] = ncaa_per_game.groupby('player_id')['year'].transform('min')
ncaa_per_game['max_year'] = ncaa_per_game.groupby('player_id')['year'].transform('max')

#add tenure
ncaa_per_game['tenure'] = ncaa_per_game['year'] - ncaa_per_game['min_year'] + 1

#Manipulate NBA data
nba_per_game['min_year'] = nba_per_game.groupby('player_id')['year'].transform('min')
nba_per_game['max_year'] = nba_per_game.groupby('player_id')['year'].transform('max')
nba_per_game['years_pro'] = nba_per_game.groupby('player_id')['year'].transform('count')

#Calculate min/max year, total years pro for NBA players
nba_plr_min_max_yr = nba_per_game.groupby('player_id').agg({'year': ['min', 'max', 'count']}).droplevel(level=0, axis=1)
nba_plr_min_max_yr.rename(columns={'min':'min_year', 'max':'max_year', 'count':'years_pro'}, inplace=True)
nba_players = nba_players.merge(nba_plr_min_max_yr, left_on='nba_player_id', right_index=True)
nba_per_game.drop(['min_year','max_year','years_pro'], axis=1, inplace=True)

nba_per_game['years_pro'] = nba_per_game.sort_values('year').groupby('player_id').cumcount()+1
nba_adv_stats['years_pro'] = nba_adv_stats.sort_values('year').groupby('player_id').cumcount()+1

#calculate median minutes/game
nba_med_grp = nba_per_game.groupby('player_id').median()
nba_players = nba_players.merge(nba_med_grp[['mp_per_g']], left_on='nba_player_id', right_index=True)
nba_players.rename(columns={'mp_per_g': 'nba_mp_per_g'}, inplace=True)

nba_adv_stats['per'] = nba_adv_stats['per'] * nba_adv_stats['mp']

#calculate mean per
nba_plr_mean_per = nba_adv_stats[(nba_adv_stats['years_pro'] < 10)].groupby('player_id').agg({'per': ['mean']}).droplevel(level=0, axis=1)
nba_plr_mean_per.rename(columns={'mean':'mean_per'}, inplace=True)
nba_players = nba_players.merge(nba_plr_mean_per, left_on='nba_player_id', right_index=True)

#################################################
#Create df of NCAA players
df = ncaa_players[['player_id', 'school_id', 'year', 'height', 'weight', 'pos']]

#Join per_game stats
df = df.merge(ncaa_per_game, on=['player_id', 'year'])

#Join advanced stats
df = df.merge(ncaa_adv[['player_id','year','ows','dws','ws','ts_pct', 'usg_pct', 'bpm', 'pprod']], on=['player_id','year'])

#Join team sos and srs
df = df.merge(ncaa_team_stats[['school_id', 'year', 'sos', 'srs']], on=['school_id', 'year'])

#add is_final_year bool
df['is_final_year'] = df['year'] == df['max_year']

#convert height to inches
df['height'] = df.apply(convert_height_to_inches, axis=1)

#Join NBA data
df = df.merge(nba_players[['college_player_id', 'years_pro', 'nba_mp_per_g', 'mean_per']].set_index('college_player_id'), how='left', left_on='player_id', right_index=True)
df[['nba_mp_per_g', 'years_pro']].fillna(value=0, axis=1, inplace=True)

#create boolean column for whether player made NBA
df['made_nba'] = (df['is_final_year'])  & (df['nba_mp_per_g'] > 0)

#assign tier
df['tier'] = df.apply(assign_tier, axis=1)

def predict_by_pos(pos, year):
    features_list = ['g', 'gs', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct', 'fg2_per_g', 'fg2a_per_g', 'fg2_pct', 'fg3_per_g', 'fg3a_per_g', 'fg3_pct', 'ft_per_g', 'fta_per_g', 'ft_pct', 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'tenure', 'height', 'weight', 'sos', 'srs', 'ows', 'dws', 'ws', 'ts_pct', 'usg_pct', 'bpm', 'pprod']
    X = df[(df['pos'] == pos) & (df['is_final_year'])]
    X = X[features_list]

    X_imp = IterativeImputer(max_iter=10).fit_transform(X)
    X = pd.DataFrame(X_imp, index=X.index, columns=X.columns)

    df.loc[X.index, X.columns] = X
    X = df[(df['is_final_year']) & (df['pos'] == pos) & (df['mp_per_g'] > 15) & (df['g'] > 25)][features_list]
    #X['per'] = (1/X['mp_per_g']) * ((X['fg_per_g'] * 85.91) + (X['stl_per_g'] * 53.897) + (X['fg3_per_g'] * 51.757) + (X['ft_per_g'] * 46.845) + (X['blk_per_g'] * 39.19) + (X['orb_per_g'] * 39.19) + (X['ast_per_g'] * 34.677) + (X['drb_per_g'] * 14.707) - (X['pf_per_g'] * 17.174) - (X['fta_per_g'] - (X['ft_per_g'])*20.091) - ((X['fga_per_g'] - X['fg_per_g'])*39.19) - (X['tov_per_g']*53.897))

    X = (X - X.min()) / (X.max() - X.min())

    predicted_to_nba = pd.DataFrame()
    for yr in range(1996, 2020):
        a = predict_make_nba(yr, X)
        predicted_to_nba = predicted_to_nba.append(a)

    ##################################################
    ##PER Regression##
    #train algorithm on players not in given year
    clf1 = SGDRegressor(alpha=.01, penalty='elasticnet')

    features_list = X.columns.tolist()

    #create dataframe of NCAA players that made NBA
    df2 = predicted_to_nba

    X2 = transform_train_data(df2[features_list])
    y2 = df2[['mean_per']].loc[X2.index]

    to_drop = list(X2.columns[X2.var() < .1])
    to_drop += ['gs']
    X2.drop(to_drop, axis=1, inplace=True)
    X2 = (X2 - X2.mean())/X2.std()

    X_new_pred = X2[df2.loc[X2.index]['year'] == year]
    X2 = X2[(df2.loc[X2.index]['year'] != year) & (df2.loc[X2.index]['year'] < 2018) & (df2.loc[X2.index]['year'] > 1995)]
    y2 = y2.loc[X2.index]

    y_new_pred = df2[['mean_per']].loc[X_new_pred.index]
    y_new_pred = (y_new_pred - y2.mean())/y2.std()
    y2 = (y2 - y2.mean())/y2.std()

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, stratify=df2.loc[y2.index]['tier'])

    clf2 = TransformedTargetRegressor(clf1)
    clf2.fit(X2_train, y2_train)

    #predict per for players in given year
    X_new_pred = X_new_pred[X2.columns.tolist()]
    new_pred = clf2.predict(X_new_pred)

    new_pred_curr_year = pd.DataFrame(new_pred, index=X_new_pred.index).merge(df.iloc[:, :-8], left_index=True, right_index=True)
    return new_pred_curr_year

predicted = pd.DataFrame()
for pos in ['G', 'F', 'C']:
    predicted = predicted.append(predict_by_pos(pos, 2019))

predicted = predicted.merge(ncaa_players[['player_id','year','player_name']], left_on=['player_id', 'year'], right_on=['player_id', 'year'])
predicted = predicted.merge(school_names, on='school_id')
predicted.drop(['player_id', 'school_id'], inplace=True, axis=1)
predicted.set_index(['player_name','school_name'], inplace=True)
predicted.sort_values(0, ascending=False, inplace=True)
predicted.rename(columns={0: 'rating'}, inplace=True)
predicted.to_csv('//Users//sidman94//Desktop//NCAA_basketball_project//d3//player_ratings.csv')
