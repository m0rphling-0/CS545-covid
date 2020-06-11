import glob, os
import pandas as pd


def process_daily_data():
    # returns a data frame that is the concatenation of all the csv files in the current directory
    # it is intended to be used to concatenate all of the us daily report covid-19 data 
    # from johns hopkins, found here: 
    # https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports_us
    df_list = []
    for file in glob.glob("*.csv"):
        df = pd.read_csv(file)
        df_list.append(df)
    main_frame = pd.concat(df_list, ignore_index=True)
    return main_frame

def separate_data(data_frame):
    date_range = "-0412-0609"
    statenames = data_frame['Province_State'].unique()
    for state in statenames:
        state_data = data_frame.loc[data_frame['Province_State'] == state]
        state_data = state_data.sort_values(by='Last_Update')
        file_name = state + date_range + ".csv"
        state_data.to_csv(file_name)

if __name__ == '__main__':
    df = process_daily_data()
    separate_data(df)
