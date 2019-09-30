#!/usr/bin/env python3
import sys
import load_ncaa_player_data
import load_ncaa_school_data
import load_nba_college_ids
import load_nba_per_game
import load_nba_adv

#year = int(sys.argv[1])
for year in range(1993, 2000):

    load_ncaa_player_data.execute(year)
    #load_ncaa_school_data.execute(year)
    #load_nba_college_ids.execute(year+1)
    #load_nba_per_game.execute(year+1)
    #load_nba_adv.execute(year+1)
