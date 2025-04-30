import os
from datetime import datetime

# Constants

# this makes it so that the outputs of the predict methods have the id as a column
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'

date_2_pred = datetime.today().replace(day=1).strftime("%Y-%m-%d")

date_2_pred_date = datetime.strptime(date_2_pred, "%Y-%m-%d")

# Subtract 20 months 
year = date_2_pred_date.year
month = date_2_pred_date.month - 21

# Adjust year and month properly
while month <= 0:
    month += 12
    year -= 1

# Final date, first day of calculated month
date_train = datetime(year, month, 1).strftime("%Y-%m-%d")

# List of 27 european countries 
eu_27_countries = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI",
                 "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL",
                 "PT", "RO", "SE", "SI", "SK"]

eu_27_countries_iso3 = ['AUT', 'BEL', 'BGR', 'CYP', 'CZE', 'DEU', 'DNK', 'EST', 'GRC', 'ESP', 'FIN',
                        'FRA', 'HRV', 'HUN', 'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'MLT', 'NLD', 'POL',
                        'PRT', 'ROU', 'SWE', 'SVN', 'SVK']

freq = "M"

# We donwload 2 more years at the beginning to deal with lags and created Nan
start_period_download = 2008
start_period_download_date = '2008-01-01'
start_period = "2010-01-01"

nb_lags = 13

max_steps_default = 2500
