# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:53:22 2021

It is not possible to fully replicate this analysis without access to Airfinity's data.
Please contact the Authors for additional information.

@author: Jrivera
"""

import os
import pandas as pd
path = os.path.dirname(__file__)
ppath = os.path.dirname(path)



def get_owid_data():
    """ Get vaccination data from Our World In Data"""
    
    try:
        url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
        return pd.read_csv(url, parse_dates = ['date'])
    
    except: 
      raise ConnectionError('Data could not be updated')
            

def add_change(df, column, new_column_name, idx=['iso_code', 'vaccine'],):
    """Add a column with the daily change, based on groups"""

    df[new_column_name] = df.groupby(
        idx)[column].transform(lambda x: x - x.shift(1))
    df.loc[df[new_column_name].isna(
    ), new_column_name] = df.loc[df[new_column_name].isna(), column]

    return df

def reshape_raw_supply_forecast(file='uk_raw_forecast_supply',name ='uk_supply_forecast'):
    """Some Airfinity data needs reshaping. Reshape for consistency"""
    
    df= pd.read_csv(path+fr'/raw_data/{file}.csv', parse_dates=['dates'])
    
    df.rename(columns = {'dates':'date'}, inplace=True)
    
    df = df.pivot_table(index=['country','date'], columns=['vaccine'])
    
    df.columns = [x[1] for x in df.columns]
    
    df.reset_index().to_excel(path+fr'/raw_data/{name}.xlsx', index=False)
    

def get_used_doses_by_manufacturer(vaccines) -> pd.DataFrame:
    """Get doses by manufacturer (manually downloded) from OWID"""

    file_name = 'covid-vaccine-doses-by-manufacturer.csv'
    

    #Add other columns if countries are administering them
    return pd.read_csv(path+rf'/raw_data/{file_name}',
                       usecols=['Code', 'Day', *vaccines],
                       parse_dates=['Day'])


def clean_used_doses_by_manufacturer(df) -> pd.DataFrame:
    """clean data on doses used by vaccine manufacturer"""
    return (df
            .rename(columns={'Code': 'iso_code', 'Day': 'date'})
            .melt(id_vars=['iso_code', 'date'],
                  var_name='vaccine',
                  value_name='cumulative_doses')
            )


def linear_projection_by_manufacturer(df: pd.DataFrame, 
                                      observed_max_date:str,
                                      days_window:int = 14,
                                      column:str ='cumulative_doses') -> pd.DataFrame:
    
    """Produce linear projections for a given column based ona given window"""

    from sklearn.linear_model import LinearRegression
    from datetime import timedelta

    import numpy as np

    #for US consistency
    original = df.loc[df.date <= observed_max_date].copy()
    
    df = original.loc[original.date >= original.date.max()-timedelta(days_window)]
    
    #Align dates
    dates_list =[]
    d_idx = pd.DataFrame({'date':pd.date_range(original.date.max()-timedelta(days_window), observed_max_date)})
    
    for iso_code in df.iso_code.unique():
        dates = df.loc[df.iso_code == iso_code].copy()
        for vax in dates.vaccine.unique():
            ds = dates.loc[dates.vaccine == vax].copy()
            ds = d_idx.merge(ds, on='date', how='left')
            ds.iso_code = iso_code
            ds.vaccine = vax
            ds = ds.set_index('date').interpolate(method='time', axis = 0, limit_direction='both')
            dates_list.append(ds.reset_index())
        
          
    df = pd.concat(dates_list, ignore_index=True)   
    
    new_dates = pd.date_range(df.date.max(), end='2021-12-31')
    forecast_dates = np.array([x.toordinal()
                               for x in new_dates.date]).reshape(-1, 1)

    forecasts_list = []

    for iso_code in df.iso_code.unique():

        df_ = df.loc[df.iso_code == iso_code].copy()

        for vax in df.vaccine.unique():

            X = df_.loc[df_.vaccine == vax].date
            X = np.array([x.toordinal() for x in X]).reshape(-1, 1)
            Y = df_.loc[df_.vaccine == vax][column].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, Y)

            forecast = model.predict(forecast_dates)

            data = pd.DataFrame(forecast, index=new_dates).reset_index()
            data.columns = ['date', column]
            data['vaccine'] = vax
            data['iso_code'] = iso_code

            forecasts_list.append(data)

    f_data = pd.concat(forecasts_list, ignore_index=True)

    return original.append(f_data, ignore_index=True).sort_values(by=['iso_code', 'vaccine', 'iso_code'])


def additional_used_by_month(df, 
                             vaccines = ['Johnson&Johnson', 
                                         'Pfizer/BioNTech',
                                         'Moderna', 
                                         'Oxford/AstraZeneca']):

    df_ = df[['iso_code', 'date', 'vaccine', 'additional_doses']].copy()

    df_.date = df_.date.dt.month

    df_ = df_.groupby(['iso_code', 'date', 'vaccine']).sum().reset_index()

    df_ = df_.pivot_table(index=['date', 'iso_code'], columns=['vaccine'])
    df_.columns = [x[1] for x in df_.columns]

    df_['Total'] = df_.sum(axis=1)

    df_ = df_.reset_index()

    df_.date = pd.to_datetime(df_.date, format='%m').dt.month_name()
    
    cols = [*['iso_code', 'date'],*vaccines, *['Total']]

    return df_[cols]


def get_owid_vaccination(indicator='people_fully_vaccinated') -> pd.DataFrame():

    return (get_owid_data()
            .filter(['iso_code', 'date', indicator])
            .assign(date=lambda x: pd.to_datetime(x.date, format='%Y-%m-%d'))
            .dropna(subset=[indicator])
            .reset_index(drop=True)
            )


def full_vaccination_forecast(vax,observed_max_date, window=14):

    vax['vaccine'] = 'All vaccines'

    vax_forecast = linear_projection_by_manufacturer(vax,
                                                     observed_max_date=observed_max_date,
                                                     days_window = window,
                                                     column='people_fully_vaccinated')
    vax_forecast = add_change(vax_forecast, 'people_fully_vaccinated', 'additional_fully_vaxxed')

    return vax_forecast.sort_values(by=['iso_code', 'date']).reset_index(drop=True)


def additional_vax_by_month(vax_forecast):

    vax_ = vax_forecast[['iso_code', 'date', 'additional_fully_vaxxed']].copy()

    vax_.date = vax_.date.dt.month

    vax_ = vax_.groupby(['iso_code', 'date']).sum().reset_index()
    vax_.date = pd.to_datetime(vax_.date, format='%m').dt.month_name()

    return vax_


#### Supply data

def get_supply(filename='us_received_airfinity', regional=False):

    cols = {'Vaccine': 'vaccine',
            'Countries that have secured doses': 'country',
            'Regions that secured these doses': 'region',
            'Total supply of COVID-19 vaccine doses secured': 'confirmed_supply',
            'Total deliveries to date of COVID-19 vaccine doses': 'delivered_supply',
            'Delivery To Date Last Updated': 'supply_date'}

    if regional == False:
        cols.pop('Regions that secured these doses')
        grouper = 'country'
    else:
        grouper = 'region'

    vax = {'AZD1222 (University of Oxford/AstraZeneca)': 'Oxford/AstraZeneca',
           'Ad26COVS1 (J&J)': 'Johnson&Johnson',
           'BNT162b2 (Pfizer/BioNTech)': 'Pfizer/BioNTech',
           'NVX-CoV2373 (Novavax)': 'Novavax',
           'Vaccine (Sanofi/GSK)': 'Sanofi/GSK',
           'mRNA-1273 (Moderna)': 'Moderna',
           'CoVLP (Medicago/GSK)': 'Medicago/GSK',
           'PTX-COVID19-B (Providence Therapeutics)':'Providence',
           'VLA2001 (Valneva/Dynavax)':'Valneva/Dynavax',
           'CVnCoV (Curevac)': 'Curevac',
           }

    df = pd.read_excel(path+fr'/raw_data/{filename}.xlsx',
                       usecols=cols.keys(), parse_dates=['Delivery To Date Last Updated'])

    df = df.rename(columns=cols)

    df.vaccine = df.vaccine.map(vax)

    return df.groupby([grouper, 'vaccine']).agg({'delivered_supply': sum,
                                                 'confirmed_supply': sum,
                                                 'supply_date': max}).reset_index(drop=False)


def get_supply_forecast(filename='us_supply_forecast', 
                        vaccines = None):

    df = pd.read_excel(path+fr'/raw_data/{filename}.xlsx',
                       parse_dates=['date'])
    
    df = df.set_index(['country','date'])
    
    df = df.filter(vaccines)
       
    
    df['Total'] = df.sum(axis=1, numeric_only=True)


    df = (df.reset_index(drop=False)
          .melt(id_vars=['country', 'date'],
                var_name='vaccine', value_name='doses')
          )

    return df.sort_values(by=['country', 'vaccine', 'date']).reset_index(drop=True)


def add_supply_change_to_forecast(df):

    return add_change(df, 'doses', 'additional_doses', idx=['country', 'vaccine'])


def additional_supply_by_month(df):
    supply = df[['country', 'date', 'vaccine', 'additional_doses']].copy()
    
    supply = supply.loc[supply.date < '2022-01-01']
    
    supply['year']= supply.date.dt.year
    supply.date = supply.date.dt.month
    

    supply = supply.groupby(['country', 'date','year', 'vaccine']).sum().reset_index()
    
    supply  = supply.pivot_table(index=['country', 'date','year'], columns=['vaccine'])
    supply.columns = [x[1] for x in supply.columns]

    
    supply = supply.reset_index()
    
    supply = supply.sort_values(by=['country','year','date']).reset_index(drop=True)
    
    supply.date = pd.to_datetime(supply.date, format='%m').dt.month_name()

    supply.year = supply.year.replace(2021,'')    
    
    supply.date = supply.date+supply.year.astype(str)
    
    supply.drop('year',axis=1, inplace=True)
    
    return supply


## donations data

def get_donations_data(filename='us_donations_airfinity'):

    cols = {'Vaccine': 'vaccine',
            'Deal type': 'type',
            'Donor': 'country',
            'Supply Number': 'donated',
            'Delivery to Date': 'delivered',
            'Delivery To Date Last Updated': 'delivery_date_last',
            }

    vaccines = {'Ad26COVS1 (J&J)': 'Johnson&Johnson',
                'BNT162b2 (Pfizer/BioNTech)': 'Pfizer/BioNTech',
                'mRNA-1273 (Moderna)': 'Moderna',
                'AZD1222 (University of Oxford/AstraZeneca)':'Oxford/AstraZeneca'}

    df = pd.read_excel(path+fr'/raw_data/{filename}.xlsx',
                       usecols=cols.keys(),
                       parse_dates=['Delivery To Date Last Updated'])

    df = df.rename(columns=cols)
    df.vaccine = df.vaccine.map(vaccines)

    df = df.loc[df.type == 'Vaccine donation']

    return df.sort_values(by='delivery_date_last').reset_index(drop=True)


def donations_to_date_summary(df):

    return df.groupby(['vaccine']).agg({'donated': sum,
                                        'delivered': sum,
                                        'delivery_date_last': max}).fillna(0).reset_index(drop=False)


def us_donation_schedule(schedule) -> pd.DataFrame:

    df = pd.DataFrame({'date': pd.date_range(
        '2021-08-01', end='2021-12-31', freq='M')})

    df.date = df.date.dt.month_name()

    
    df_list = []

    for vax in schedule.keys():
        d2 = df.copy()
        d2['vaccine'] = vax
        df_list.append(d2)

    df = pd.concat(df_list, ignore_index=True)

    for vax in schedule.keys():

        df.loc[df.vaccine == vax,
               'delivered_forecast'] = df.date.map(schedule[vax])

    return df


def can_donation_schedule() -> pd.DataFrame:

    df = pd.DataFrame({'date': pd.date_range(
        '2021-08-01', end='2021-12-31', freq='M')})

    df.date = df.date.dt.month_name()

    doses = {'Pfizer/BioNTech': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Moderna':         {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Johnson&Johnson': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Oxford/AstraZeneca': {'August': 0,
                                    'September': 0,
                                    'October': 0,
                                    'November': 0,
                                    'December': 0},
             'All_vaccines':         {'August': 0*1e6,
                                      'September': 13*1e6,
                                      'October': 0*1e6,
                                      'November': 0*1e6,
                                      'December': 0*1e6},
             }
    df_list = []

    for vax in doses.keys():
        d2 = df.copy()
        d2['vaccine'] = vax
        df_list.append(d2)

    df = pd.concat(df_list, ignore_index=True)

    for vax in doses.keys():

        df.loc[df.vaccine == vax,
               'delivered_forecast'] = df.date.map(doses[vax])

    return df

def uk_donation_schedule() -> pd.DataFrame:

    df = pd.DataFrame({'date': pd.date_range(
        '2021-08-01', end='2021-12-31', freq='M')})

    df.date = df.date.dt.month_name()

    doses = {'Pfizer/BioNTech': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Moderna':         {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Johnson&Johnson': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Oxford/AstraZeneca': {'August': 0,
                                    'September': 0,
                                    'October': 0,
                                    'November': 0,
                                    'December': 0},
             'All_vaccines':         {'August': 0,
                                      'September': 5*1e6,
                                      'October': 8.333333*1e6,
                                      'November': 8.333333*1e6,
                                      'December': 8.333334*1e6},
             }
    df_list = []

    for vax in doses.keys():
        d2 = df.copy()
        d2['vaccine'] = vax
        df_list.append(d2)

    df = pd.concat(df_list, ignore_index=True)

    for vax in doses.keys():

        df.loc[df.vaccine == vax,
               'delivered_forecast'] = df.date.map(doses[vax])

    return df


def jpn_donation_schedule() -> pd.DataFrame:

    df = pd.DataFrame({'date': pd.date_range(
        '2021-08-01', end='2021-12-31', freq='M')})

    df.date = df.date.dt.month_name()

    doses = {'Pfizer/BioNTech': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Moderna':         {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Johnson&Johnson': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Oxford/AstraZeneca': {'August': 0,
                                    'September': 2*1e6,
                                    'October': 2*1e6,
                                    'November': 0,
                                    'December': 0},
             'All_vaccines':         {'August': 0,
                                      'September': 0,
                                      'October': 0*1e6,
                                      'November':0*1e6,
                                      'December':0*1e6},
             }
    df_list = []

    for vax in doses.keys():
        d2 = df.copy()
        d2['vaccine'] = vax
        df_list.append(d2)

    df = pd.concat(df_list, ignore_index=True)

    for vax in doses.keys():

        df.loc[df.vaccine == vax,
               'delivered_forecast'] = df.date.map(doses[vax])

    return df


def ita_donation_schedule() -> pd.DataFrame:

    df = pd.DataFrame({'date': pd.date_range(
        '2021-08-01', end='2021-12-31', freq='M')})

    df.date = df.date.dt.month_name()

    doses = {'Pfizer/BioNTech': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Moderna':         {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Johnson&Johnson': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Oxford/AstraZeneca': {'August': 0,
                                    'September': 0*1e6,
                                    'October': 0*1e6,
                                    'November': 0,
                                    'December': 0},
             'All_vaccines':         {'August': 0,
                                      'September': 0,
                                      'October': 1.875*1e6,
                                      'November':1.875*1e6,
                                      'December':1.875*1e6},
             }
    df_list = []

    for vax in doses.keys():
        d2 = df.copy()
        d2['vaccine'] = vax
        df_list.append(d2)

    df = pd.concat(df_list, ignore_index=True)

    for vax in doses.keys():

        df.loc[df.vaccine == vax,
               'delivered_forecast'] = df.date.map(doses[vax])

    return df


def fra_donation_schedule() -> pd.DataFrame:

    df = pd.DataFrame({'date': pd.date_range(
        '2021-08-01', end='2021-12-31', freq='M')})

    df.date = df.date.dt.month_name()

    doses = {'Pfizer/BioNTech': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Moderna':         {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Johnson&Johnson': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Oxford/AstraZeneca': {'August': 0,
                                    'September': 0*1e6,
                                    'October': 0*1e6,
                                    'November': 0,
                                    'December': 0},
             'All_vaccines':         {'August': 0,
                                      'September': 0,
                                      'October': 3.75*1e6,
                                      'November':3.75*1e6,
                                      'December':3.75*1e6},
             }
    df_list = []

    for vax in doses.keys():
        d2 = df.copy()
        d2['vaccine'] = vax
        df_list.append(d2)

    df = pd.concat(df_list, ignore_index=True)

    for vax in doses.keys():

        df.loc[df.vaccine == vax,
               'delivered_forecast'] = df.date.map(doses[vax])

    return df


def ger_donation_schedule() -> pd.DataFrame:

    df = pd.DataFrame({'date': pd.date_range(
        '2021-08-01', end='2021-12-31', freq='M')})

    df.date = df.date.dt.month_name()

    doses = {'Pfizer/BioNTech': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Moderna':         {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Johnson&Johnson': {'August': 0,
                                 'September': 0,
                                 'October': 0,
                                 'November': 0,
                                 'December': 0},

             'Oxford/AstraZeneca': {'August': 0,
                                    'September': 0*1e6,
                                    'October': 0*1e6,
                                    'November': 0,
                                    'December': 0},
             'All_vaccines':         {'August': 0,
                                      'September': 0,
                                      'October': 0*1e6,
                                      'November':0*1e6,
                                      'December':0*1e6},
             }
    df_list = []

    for vax in doses.keys():
        d2 = df.copy()
        d2['vaccine'] = vax
        df_list.append(d2)

    df = pd.concat(df_list, ignore_index=True)

    for vax in doses.keys():

        df.loc[df.vaccine == vax,
               'delivered_forecast'] = df.date.map(doses[vax])

    return df

def donations_forecast_summary(df):
    
    order= {'August':1, 'September':2, 'October':3,'November':4,'December':5}
    
    
    df = df.pivot_table(index=['date'], columns = 'vaccine')
    df.columns = [x[1] for x in df.columns]
    
    df = df.reset_index(drop=False)
    df['ord'] = df.date.map(order)
    df.sort_values('ord',inplace=True)
    df.drop('ord', axis=1, inplace=True)
    
    df = df.set_index('date')
    
    df['Total'] = df.sum(axis=1)
    df['Cumulative'] = df.Total.cumsum()
    
    
    
    return df.reset_index(drop=False)

def format_workbook(writer, wb, ws, textcols=['A:C'], numcols=['D:F']):
    fmt_number = wb.add_format({'num_format':'#,##0'})
    
    ws.set_column(textcols, 20)
    ws.set_column(numcols, 22, fmt_number)
    


        
def export_excel(file_name, worksheet_name,
                 dataframes_dict):
    
    writer = pd.ExcelWriter(r'C:\Users\jrivera\OneDrive - THE ONE CAMPAIGN\supply_demand'+rf'/{file_name}.xlsx', engine = 'xlsxwriter')
    
    workbook = writer.book
    
    bold = workbook.add_format({'bold':True})
    
    worksheet = workbook.add_worksheet(worksheet_name)
    writer.sheets[worksheet_name] = worksheet
    
    dflen= 0
    
    for i, t in enumerate(dataframes_dict.items()):
        
        name = t[0]
        shape = t[1].shape[0]
        #dflen = dflen+(i*4)
        
        worksheet.write_string(dflen,0, name, bold)
        t[1].to_excel(writer, sheet_name = worksheet_name, index=False, startrow=dflen+1)
        
        dflen += shape+4
        
    format_workbook(writer, workbook, worksheet, textcols= 'A:B', numcols = 'C:M')
    
    writer.save()
    
    
        
        
        
