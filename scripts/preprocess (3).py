
import os
import sys 
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

# pre-processing
def pp_admissions(sparkjson):
    """Filter the json file read by PySpark
    
    Take only admitted patients. Get rid of PUI. 
    
    Arguments:
        sparjson: object spark.read.json(fpath), where spark = (SparkSession.builder.appName("ED_process").getOrCreate())
        
    Returns:
        pd.DataFrame: processed admisisons
    """
    
    df = sparkjson.toPandas()
    df = df.loc[df['admitted']==1,:]
    df = df.drop(columns=['admitted', 'pui'])
    
    # convert to datetime
    df['discharge_datetime'] = pd.to_datetime(df['discharge_datetime'])
    df['ed_arrival_datetime'] = pd.to_datetime(df['ed_arrival_datetime'])
    df['min_order_ts'] = pd.to_datetime(df['min_order_ts'])
    
    # NOTE:
    # admit_df.duplicated().sum()==0
    
    return df

# filter demographics 
def pp_demographics_OLD(sparkjson, admit_df):
    """Retain demographics for admitted patients. 
    
    Filters by patient ID.
    
    Args:
        sparkjson: spark.read.json(fpath) for demographics data
        admit_df (pd.DataFrame): addmitted patients filtered 
        
    Returns:
        pd.DataFrame of demographcs
        
    """
    demo_df = sparkjson.toPandas()
    demo_df = demo_df.loc[demo_df['person_id'].isin(admit_df['person_id']),:]
    
    # recode ethnicity NA to unknown 
    demo_df['ethnicity'].fillna('Unknown', inplace=True)
    demo_df['ethnicity'].replace('No matching concept', 'Unknown', inplace=True)
    
    # recod race to NA -> unknown and 4 categories 
    demo_df['race'].replace('Asian', 'Other', inplace=True)
    demo_df['race'].replace('American Indian or Alaska Native', 'Other', inplace=True)
    demo_df['race'].replace('No matching concept', 'Unknown', inplace=True)
    demo_df['race'].replace('Other Race', 'Other', inplace=True)
    demo_df['race'].replace('Race not stated', 'Unknown', inplace=True)
    demo_df['race'].replace('Native Hawaiian or Other Pacific Islander', 'Other', inplace=True)
    demo_df['race'].fillna('Unknown', inplace=True)
    
    # covert datetime
    demo_df['min_order_ts'] = pd.to_datetime(demo_df['min_order_ts'])
    demo_df['order_ts'] = pd.to_datetime(demo_df['order_ts'])
    demo_df['resulted_ts'] = pd.to_datetime(demo_df['resulted_ts'])
    
    # drop duplicates
    demo_df = demo_df.drop_duplicates()
    
    return demo_df

def pp_demographics(df):
    """Retain demographics for admitted patients. 
    
    Replaces race and ethnicity categories. 
    
    Args:
        df (pd.DataFrame): loaded from csv shared by AT
        
    Returns:
        pd.DataFrame of demographcs
        
    """
    if False:
        # load from csv, if True, make sure to update arg
        ## NOTE: had roblems with converting persion_id to int64
        df = sparkjson.toPandas()
    df['person_id'].dropna(inplace=True)
    df['person_id'] = df['person_id'].astype('int64')
    
    # recode ethnicity NA to unknown 
    df['ethnicity'].fillna('Unknown', inplace=True)
    df['ethnicity'].replace('No matching concept', 'Unknown', inplace=True)
    
    # recod race to NA -> unknown and 4 categories 
    df['race'].replace('Asian', 'Other', inplace=True)
    df['race'].replace('American Indian or Alaska Native', 'Other', inplace=True)
    df['race'].replace('No matching concept', 'Unknown', inplace=True)
    df['race'].replace('Other Race', 'Other', inplace=True)
    df['race'].replace('Race not stated', 'Unknown', inplace=True)
    df['race'].replace('Native Hawaiian or Other Pacific Islander', 'Other', inplace=True)
    df['race'].fillna('Unknown', inplace=True)
    
    # covert datetime
    df['min_order_ts'] = pd.to_datetime(df['min_order_ts'])
    df['order_ts'] = pd.to_datetime(df['order_ts'])
    df['resulted_ts'] = pd.to_datetime(df['resulted_ts'])
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def pp_labs(sparkjson, time_interval=120):
    """Function to preprocess labs.
    
    Steps:
        1. selects labs of base cohort through join
        2. replaces non-numerical values
        3. filters based on time interval
        4. selects first value within filtered time interval
        5. pivots lab types into columns

    Parameters:
        labs: dataframe of labs
        base: dataframe of base cohort
        time_interval: time interval/window to be chosen from ed arrival to lab measurement
    
    Returns: 
        dataframe: a dataframe with each lab in a column as numerical type
    """
    # join with base cohort
    if False:
        # don't join, do separately later in merge_pp()
        #labs = labs.merge(base, how='inner', on='visit_occurrence_id') 
        continue 
        
    # select relevant columns 
    labs = sparkjson.toPandas()
    labs = labs[['lab_name',
                 'ed_arrival_datetime',
                 'measurement_datetime',
                 'value_source_value',
                 'visit_occurrence_id']]
    
    # replace non-numeric characters with "" for values
    labs['value_source_value'] = labs['value_source_value'].str.replace('[^\d.]+', '')
    
    # change datetime types
    labs['ed_arrival_datetime'] = pd.to_datetime(labs['ed_arrival_datetime'])
    labs['measurement_datetime'] = pd.to_datetime(labs['measurement_datetime'])
    
    # create variable for time interval
    labs['time_diff'] = (labs['measurement_datetime']-labs['ed_arrival_datetime']).astype('timedelta64[m]')
    
    #get distinct values
    labs = labs.drop_duplicates(subset=('lab_name',
                                        'measurement_datetime',
                                        'visit_occurrence_id'))
    
    #filter labs base on time window
    labs = labs[labs['time_diff']<time_interval]
    
    #group by lab_name and visit_occurrence_id and get first value
    labs = labs[['lab_name', 'visit_occurrence_id', 'value_source_value']]
    labs = (labs
          .groupby(['visit_occurrence_id', 'lab_name'], as_index=False)
          .aggregate(lambda x: list(x)[0]))
    
    #pivot data
    labs = labs.pivot(index='visit_occurrence_id',
                      columns = 'lab_name',
                      values='value_source_value')
    
    for i in labs.columns:
        labs.loc[:,i] = pd.to_numeric(labs.loc[:,i], errors='raise')
    return labs



def add_header(outcomes, outcomes_header):
    """Put header on outcomes file.
    
    Parameters:
        outcomes: sparkjson obj of outcomes
        outcomes_header: sparkjson obj of outcomes_header

    Returns: 
        dataframe: a dataframe of outcomes with appropriate header
    """
    # note:  csn is included in header but not in outcomes file
    outcomes = outcomes.toPandas()
    outcomes_header = outcomes_header.toPandas()
    
    outcomes.columns=outcomes_header.values.tolist()[0][0:15]
    
    return outcomes

def pp_outcomes(outcomes, ex_time_interval=240, outcome_time_interval=1440):
    """Processes outcomes into final form
    
    TODO: 
        resolve SettingWithCopyWarning by isolating line and changing to `.loc[row_indexer,col_indexer] = value` instead

    Steps:
        1. Create variables(timestamp, timediff) for first critical illness state
        2. Create exclusion flag based on critical illness state < 4 hours
        3. Replace timestamps with timediff values
        4. 
        
    Parameters:
        outcomes: pnadas dataframe of outcomes

    Returns: 
        dataframe: a pandas dataframe of outcomes
        creates two new variables
        'primary_yn' : primary outcome
        'exflag_yn' : flag for exclusion if outcome occurred within first 4 hours
        
    Assumptions:  
        if no timestamp for any o2 device assume room air
    """
    # restrict to outcomes with visit occurrence ids and reset type
    outcomes = outcomes.dropna(subset=['visit_occurrence_id'])
    outcomes['visit_occurrence_id'] = outcomes['visit_occurrence_id'].astype('int64')
    
    #subset data to only needed variables
    #create death timestamp (if discharge is expired replace with visit_end_datetime)
    outcomes['death'] = (np.select(
        [
            outcomes['discharge_to_source_value']=='Expired'
        ],
        [
            outcomes['visit_end_datetime']
        ],
        default=None))
  
    #subset data to only needed variables 
    outcomes = outcomes.drop(['person_id',
                              'order_ts',
                              'received_ts',
                              'resulted_ts',
                              'result',
                              'discharge_to_source_value'], axis=1) 
  
    # drop visit end datetime
    outcomes = outcomes.drop(['visit_end_datetime'], axis=1)
  
    # gather data
    outcomes = pd.melt(outcomes, id_vars=['visit_occurrence_id', 'visit_start_datetime'])
    outcomes = outcomes.dropna()
    outcomes['visit_start_datetime'] = pd.to_datetime(outcomes['visit_start_datetime'])
    outcomes['value'] = pd.to_datetime(outcomes['value'], errors='coerce')
    
    # create variable for time interval
    outcomes['time_diff'] =(outcomes['value']-outcomes['visit_start_datetime']).astype('timedelta64[m]')
    outcomes = outcomes.pivot(index='visit_occurrence_id',
                              columns = 'variable',
                              values='time_diff')
    outcomes['exflag_yn'] = (np.select(
        [
            ((outcomes['HIGH_FLOW'] < ex_time_interval) 
             & ((outcomes['HIGH_FLOW'] > outcomes['LOW_FLOW']) 
                | (outcomes['HIGH_FLOW'] > outcomes['ROOM_AIR']))
            ),
            ((outcomes['NON_INVASIVE'] < ex_time_interval) 
             & ((outcomes['NON_INVASIVE'] > outcomes['LOW_FLOW']) 
                | (outcomes['NON_INVASIVE'] > outcomes['ROOM_AIR']))
            ),
            ((outcomes['INVASIVE'] < ex_time_interval) 
             & ((outcomes['INVASIVE'] > outcomes['LOW_FLOW'])
                | (outcomes['INVASIVE'] > outcomes['ROOM_AIR']))
            ),
            ((outcomes['death'] < ex_time_interval) 
             & ((outcomes['death'] > outcomes['LOW_FLOW']) 
                | (outcomes['death'] > outcomes['ROOM_AIR']))
            )
        ],
        [
            1,
            1,
            1,
            1,
        ],
        default=0))
    
    outcomes['primary_outcome_yn'] = (np.select(
        [
            outcomes['HIGH_FLOW'] < outcome_time_interval,
            outcomes['NON_INVASIVE'] < outcome_time_interval,
            outcomes['INVASIVE'] < outcome_time_interval,
            outcomes['death'] < outcome_time_interval
        ],
        [
            1,
            1,
            1,
            1,
        ],
        default=0))
  
    #unstack and reset index
    outcomes = outcomes.reset_index()
    return outcomes

def pp_edextrajson(sparkjson):
    """Process outcomes from ed_extra file from json.
    
    Args:
        sparkjson: object spark.read.json(fpath) for fpath pointing to vitals json
        
    Returns:
        pd.DataFrame
    
    """
    ed = sparkjson.toPandas()
    ed['visit_occurrence_id'] = ed['visit_occurrence_id'].astype('int64')
    
    if ed.duplicated().sum() != 0:
        print('ADD filtering for duplicates')
        
    # drop duplicates in ed_extra on visit occureence
    ed = ed.drop_duplicates(subset='visit_occurrence_id')
    
    # replacements
    ed['AcuityLevel'].fillna('Unknown', inplace=True)
    ed['AcuityLevel'].replace('*Unspecified', 'Unknown')
    
    # datetime
    ed['DepartureInstant'] = pd.to_datetime(ed['DepartureInstant'])
    
    for category in ed['FinancialClass'].unique():
        if category=='Medicaid Managed Care':
            ed['FinancialClass'].replace(category, 'Medicaid', inplace=True)
        elif category=='Medicare Managed Care':
            ed['FinancialClass'].replace(category, 'Medicare', inplace=True)
        elif category=='Managed Care':
            ed['FinancialClass'].replace(category, 'Other', inplace=True)
        elif category=='BCBS':
            ed['FinancialClass'].replace(category, 'Commercial', inplace=True)
        elif 'Worker' in category:
            ed['FinancialClass'].replace(category, 'Other', inplace=True)
        ed['FinancialClass'].fillna('Unknown', inplace=True)
        
        for category in ed['PreferredLanguage'].unique():
            if category=='*Unspecified':
                ed['PreferredLanguage'].replace(category, 'Unknown', inplace=True)
            elif category=='English' or category=='Unknown':
                continue
            else:
                ed['PreferredLanguage'].replace(category, 'NotEnglish', inplace=True)
        ed['PreferredLanguage'].fillna('Unknown', inplace=True)
        
        ed['SmokingStatus'].replace('Never Assessed', 'Unknown', inplace=True)
        ed['SmokingStatus'].replace('Never Smoker ', 'Never Smoker', inplace=True)
        ed['SmokingStatus'].replace('*Unknown', 'Unknown', inplace=True)
        ed['SmokingStatus'].replace('Unknown If Ever Smoked', 'Unknown', inplace=True)
        ed['SmokingStatus'].fillna('Unknown', inplace=True)
        
        # top 10 complaints
        for i in ed['chief_complaint'].unique():
            if 'FEVER' in i:
                ed['chief_complaint'].replace(i, 'FEVER', inplace=True)
        top10 = ed.groupby('chief_complaint').count().sort_values(['FinancialClass'], ascending=False).iloc[0:10,:].index.to_list()
        
        for i in ed['chief_complaint'].unique():
            if i in top10:
                continue
            else:
                ed['chief_complaint'].replace(i, 'Other', inplace=True)
    
    return ed

def pp_ed_extra(ed):
    """Process file shared by AT.
    
    Arguments:
        ed (pd.DataFrame): loaded from csv
    
    Returns:
        pd.DataFrame
        
    """
    
    ed = ed.reset_index()
    ed = ed.rename(columns={'PatientDurableKey':'person_id',
                            'EncounterKey':'visit_occurrence_id',
                            'Name':'chief_complaint'})
    
    
    # drop duplicates in ed_extra on visit occureence
    ed = ed.drop_duplicates(subset='visit_occurrence_id')
    
    # replacements
    ed['AcuityLevel'].fillna('Unknown', inplace=True)
    ed['AcuityLevel'].replace('*Unspecified', 'Unknown')
    
    # datetime
    ed['DepartureInstant'] = pd.to_datetime(ed['DepartureInstant'])
    
    for category in ed['FinancialClass'].unique():
        if category=='Medicaid Managed Care':
            ed['FinancialClass'].replace(category, 'Medicaid', inplace=True)
        elif category=='Medicare Managed Care':
            ed['FinancialClass'].replace(category, 'Medicare', inplace=True)
        elif category=='Managed Care':
            ed['FinancialClass'].replace(category, 'Other', inplace=True)
        elif category=='BCBS':
            ed['FinancialClass'].replace(category, 'Commercial', inplace=True)
        elif 'Worker' in category:
            ed['FinancialClass'].replace(category, 'Other', inplace=True)
        ed['FinancialClass'].fillna('Unknown', inplace=True)
        
        for category in ed['PreferredLanguage'].unique():
            if category=='*Unspecified':
                ed['PreferredLanguage'].replace(category, 'Unknown', inplace=True)
            elif category=='English' or category=='Unknown':
                continue
            else:
                ed['PreferredLanguage'].replace(category, 'NotEnglish', inplace=True)
        ed['PreferredLanguage'].fillna('Unknown', inplace=True)
        
        ed['SmokingStatus'].replace('Never Assessed', 'Unknown', inplace=True)
        ed['SmokingStatus'].replace('Never Smoker ', 'Never Smoker', inplace=True)
        ed['SmokingStatus'].replace('*Unknown', 'Unknown', inplace=True)
        ed['SmokingStatus'].replace('Unknown If Ever Smoked', 'Unknown', inplace=True)
        ed['SmokingStatus'].fillna('Unknown', inplace=True)
        
        # top 10 complaints
        for i in ed['chief_complaint'].unique():
            if 'FEVER' in i:
                ed['chief_complaint'].replace(i, 'FEVER', inplace=True)
        top10 = ed.groupby('chief_complaint').count().sort_values(['FinancialClass'], ascending=False).iloc[0:10,:].index.to_list()
        
        for i in ed['chief_complaint'].unique():
            if i in top10:
                continue
            else:
                ed['chief_complaint'].replace(i, 'Other', inplace=True)
    
    
    return ed


def load_csv(filename):
    """Load data stored in csv files. 
    
    Returns:
        pd.DataFrame
    """
    return pd.read_csv(filename, index_col=0)

def pp_vitals(df):
    """Pre-process vitals dataframe
    
    Adds two features (Mean Arterial Pressure and Heart Index) and
    takes min max median first and last vitals measurement per 
    visit_occurence_id
    
    Arguments:
        df (pd.DataFrame): vitals df loaded from `load_vitals()`
        
    Returns:
        pd.DataFrame: pre-processed vitals, ready for integration
    """
    
    df = df.dropna(subset=['visit_occurrence_id'])
    df['ed_arrival_datetime'] = pd.to_datetime(df['ed_arrival_datetime'])
    df['measurement_datetime'] = pd.to_datetime(df['measurement_datetime'])
    
    df['visit_occurrence_id'] = df['visit_occurrence_id'].astype('int64')
    
    # drop complete duplicates
    df = df.drop_duplicates()
    
        
        # NOTE: 
        ## x.duplicated().sum() != 0 # because some vitals get measured at the same time
    
    
    # drop vial if >4h after ed_arrival_datetime
    df['time_diff'] =(df['measurement_datetime']-df['ed_arrival_datetime']).astype('timedelta64[h]')
    df = df.loc[df['time_diff']<=4,:]
    df = df.drop(columns='time_diff')
    
    
    
    # long --> wide, filter values across rows, wide --> long
    x = (df.drop(columns=['visit_occurrence_id','ed_arrival_datetime','measurement_datetime']).pivot(columns='vital_name'))
    x[['visit_occurrence_id','ed_arrival_datetime','measurement_datetime']] = df.loc[:,['visit_occurrence_id','ed_arrival_datetime','measurement_datetime']]
    x.columns = [x.columns.get_level_values(0)[i] if second=='' else second for i,second in enumerate(x.columns.get_level_values(1))]
    
    # filter physiological values
    ## SBP >300 or <30 
    x.loc[(x.loc[:,'SBP']<30) | 
          (x.loc[:,'SBP']>300),'SBP']=np.nan
    ## DBP >200 or <20 
    x.loc[(x.loc[:,'DBP']<20) | 
          (x.loc[:,'DBP']>200),'DBP']=np.nan
    ## HR >300 or <20 
    x.loc[(x.loc[:,'HR']<20) | 
          (x.loc[:,'HR']>300),'HR']=np.nan
    ## RR >60 or <5 
    x.loc[(x.loc[:,'RR']<5) | 
          (x.loc[:,'RR']>60),'RR']=np.nan
    ## SpO2 >100 or <40 
    x.loc[(x.loc[:,'SPO2']<40) | 
          (x.loc[:,'SPO2']>100),'SPO2']=np.nan
    ## Temp >106 or <80 
    x.loc[(x.loc[:,'TEMP']<80) | 
          (x.loc[:,'TEMP']>106),'TEMP']=np.nan
    ## BMI <10 or >80 
    x.loc[(x.loc[:,'BMI']<10) | 
          (x.loc[:,'BMI']>80),'BMI']=np.nan
    x = x.melt(id_vars=['visit_occurrence_id','ed_arrival_datetime','measurement_datetime'],
               value_name='value_source_value',var_name='vital_name') 

    # calculate MAP+SI after aggregating, then append rows
    def MAP(x_sub):
        sbp = x_sub.loc[x_sub.loc[:,'vital_name']=='SBP', 'value_source_value']
        dbp = x_sub.loc[x_sub.loc[:,'vital_name']=='DBP', 'value_source_value']
        if not any(sbp.notna()):
            sbp = np.nan
        else:
            sbp = np.where(sbp.notna())[0][0] # take first measurement
        if not any(dbp.notna()):
            dbp = np.nan
        else:
            dbp = np.where(dbp.notna())[0][0] # take first measurement
        return (1/3)*(sbp + 2*dbp) 
    def SI(x_sub):
        sbp = x_sub.loc[x_sub.loc[:,'vital_name']=='SBP', 'value_source_value']
        hr = x_sub.loc[x_sub.loc[:,'vital_name']=='HR', 'value_source_value']
        if not any(sbp.notna()):
            sbp = np.nan
        else:
            sbp = np.where(sbp.notna())[0][0] # take first measurement
        if not any(hr.notna()):
            hr = np.nan
        else:
            hr = np.where(hr.notna())[0][0] # take first measurement
        return hr / sbp
    
    y = x.groupby(['visit_occurrence_id','ed_arrival_datetime','measurement_datetime']).apply(MAP)
    y = y.reset_index()
    y.rename(columns={0:'value_source_value'}, inplace=True)
    y['vital_name']=['MAP']*y.shape[0]
    df = df.append(y)
    del y
    
    y = x.groupby(['visit_occurrence_id','ed_arrival_datetime','measurement_datetime']).apply(SI)
    y = y.reset_index()
    y.rename(columns={0:'value_source_value'}, inplace=True)
    y['vital_name']=['SI']*y.shape[0]
    df = df.append(y)
    del x,y
    
    # get first and last measurement per vital
    # sort by first
    df = df.sort_values(by=['measurement_datetime'])
    x = (df.groupby(['visit_occurrence_id','ed_arrival_datetime','vital_name'])
         .agg(median=pd.NamedAgg(column='value_source_value', aggfunc='median'),
              min=pd.NamedAgg(column='value_source_value', aggfunc='min'),
              max=pd.NamedAgg(column='value_source_value', aggfunc='max'),
              first = pd.NamedAgg(column='value_source_value', 
                                  aggfunc=lambda x:x.iloc[0]),
              last = pd.NamedAgg(column='value_source_value', 
                                  aggfunc=lambda x:x.iloc[-1]), # NOTE: may keep if NaN
              n_measurement=pd.NamedAgg(column='value_source_value', aggfunc='count'),
              all_measurements=pd.NamedAgg(column='value_source_value',
                                           aggfunc=lambda x:list(x)))) # ordered by measurement_time
    x = x.unstack()
    x.columns = ['{}_{}'.format(first,x.columns.get_level_values(1)[i]) for i,first in enumerate(x.columns.get_level_values(0))]
    df = x
    del x
    
    # drop ESI
    esi_cols = [i for i in df.columns if 'ESI' in i]
    df.drop(columns=esi_cols)
    
    # un-multiindex
    df = df.reset_index()
    
    return df


def pp_meds(meds):
    """Pre-process shared meds file.
    
    Returns:
        pd.DataFrame: pre-processed for model dev
    """
    meds = meds.reset_index()
    meds = meds.drop(columns=['outpt_med_yn', 'person_id']) # don't need PID, just slows things down
    # drop entire duplicates
    meds = meds.drop_duplicates() # is 0, but forseeably with new data may not be
    meds = pd.get_dummies(meds, columns=['PharmaceuticalClass']).groupby('visit_occurrence_id').sum() # why is this slow?
    meds = meds.reset_index()
    
    return meds

def pp_xrays(xrays):
    """Convert the label to string, keep only visit_occurence_id for merge.
    
    Returns:
        pd.DataFrame: pre-processed for model dev
    """
    xrays = xrays.reset_index()
    xrays['cxr_class'] = xrays['cxr_class'].astype(str)
    
    if xrays['visit_occurrence_id'].duplicated().sum() > 0:
        print('need to filter duplicates') # could raise warning but don't be annoying
    
    return xrays


def pp_site(site):
    """Assumes site is a dataframe but preprocesses this dataframe lightly
    
    Returns:
        pd.DataFrame
    """
    site = site.reset_index()
    if site['visit_occurrence_id'].duplicated().sum()>0:
        print('duplicates find. Add drop')
    
    return site


def merge_pp(outcomes, demographics, 
             ed_extra, labs, vitals, 
             meds, xrays, site, 
             verbose=False, 
             check_right_duplicates=False):
    """Left join pre-processed dataframes on outcomes dataframe.
    
    # TODO: export merged to feather/csv/pickle for fast loading 
    # TODO: dictionary of processed variables by category
    
    Arguments:
        pd.DataFrame: all are preprocessed dataframes
        
    Reutnrs:
        pd.DataFrame: data to use for modeling (no variable transforms applied)
    
    """
    
    df = outcomes
    
    # merge
    if verbose:
        print('Merging pre-processed dataframes\n  {}-rows in outcomes'.format(df.shape[0]))
    if check_right_duplicates:
        if ed_extra['visit_occurrence_id'].duplicated().sum()>0:
            print('    NOTE: RHS of df to join has visit_occurence_id duplicates')
    df = df.merge(ed_extra, left_on='visit_occurrence_id', right_on='visit_occurrence_id', how='left')
    if verbose:
        print('  {}-rows after merging ed_extra'.format(df.shape[0]))
    if check_right_duplicates:
        if demographics['person_id'].duplicated().sum()>0:
            print('   NOTE: RHS of df to join has person_id duplicates')
    df = df.merge(demographics, left_on='person_id', right_on='person_id', how='left')
    if verbose:
        print('  {}-rows after merging demographics'.format(df.shape[0]))
    if check_right_duplicates:
        if labs.index.duplicated().sum()>0:
            print('   NOTE: RHS of df to join has visit_occurence_id duplicates')
    df = df.merge(labs, left_on='visit_occurrence_id', right_index=True, how='left')
    if verbose:
        print('  {}-rows after merging labs'.format(df.shape[0]))
    if check_right_duplicates:
        if vitals['visit_occurrence_id'].duplicated().sum()>0:
            print('    NOTE: RHS of df to join has visit_occurence_id duplicates')
    df = df.merge(vitals, left_on='visit_occurrence_id', right_on='visit_occurrence_id', how='left')
    if verbose:
        print('  {}-rows after merging vitals'.format(df.shape[0]))
    if check_right_duplicates:
        if meds['visit_occurrence_id'].duplicated().sum()>0:
            print('    NOTE: RHS of df to join has visit_occurence_id duplicates')
    df = df.merge(meds, left_on='visit_occurrence_id', right_on='visit_occurrence_id', how='left')
    if verbose:
        print('  {}-rows after merging meds'.format(df.shape[0]))
    if check_right_duplicates:
        if xrays['visit_occurrence_id'].duplicated().sum()>0:
            print('RHS of df to join has visit_occurence_id duplicates')
    df = df.merge(xrays, left_on='visit_occurrence_id', right_on='visit_occurrence_id', how='left')
    if check_right_duplicates:
        if site['visit_occurrence_id'].duplicated().sum()>0:
            print('    NOTE: RHS of df to join has visit_occurence_id duplicates')
    if verbose:
        print('  {}-rows after merging xrays'.format(df.shape[0]))
    df = df.merge(site, left_on='visit_occurrence_id', right_on='visit_occurrence_id', how='left')
    if verbose:
        print('  {}-rows after merging site data'.format(df.shape[0]))
        
    # filter
    df = df.query('exflag_yn==0 & age_years>17')
    if verbose:
        print('\nAfter filtering, data has {}-rows'.format(df.shape[0]))
    
    # drop duplicates?
    if verbose:
        print('\nN visit_occurence_id duplicates: {}'.format(df['visit_occurrence_id'].duplicated().sum()))
    df = df.drop_duplicates(subset='visit_occurrence_id')
    
    if verbose:
        print('\nFinal nrows={}'.format(df.shape[0]))
        
    return df

def vars2dict(dfs, save=False):
    """Get dictionary of variables to use in model dev.
    
    Arguments:
        dfs (dict): dictionary of dataframes where key is name for df, 
            value is dataframe
        save (bool): default=False. If true, give a filename to save csv.
    
    Returns: 
        pd.DataFrame: list of variales to use in model 
    """
    variables = {}
    dtypes = {}
    for k,df in dfs.items():
        cols = []
        for i in df.columns:
            if (i not in ['visit_occurrence_id','person_id','measurement',
                          'resulted_ts','min_order_ts','icu_admit_instant']):
                cols.append(i)
                dtypes[i] = df.loc[:,i].dtype.name
        variables[k] = cols
    
    vardf = pd.DataFrame()
    vardf['Variable'] = [v for k,vlist in variables.items() for v in vlist]
    vardf['Source'] = [j for i in [[k]*len(v) for k,v in variables.items()] for j in i]
    vardf['dtype'] = vardf['Variable'].map(dtypes)
    vardf['use1_transform2'] = 1
    
    vardf.loc[[True if 'all_measurements' in i else False for i in vardf['Variable']],'use1_transform2'] = 0
    
    if save is not None:
        vardf.to_csv(save)
        
    return vardf


def vars2keep(annotated_csv):
    """Which variables to keep, parsed into a dictionary.
    
    Arguments:
        annotated_csv (str): filename of annotated data dictionary
        
    Returns:
        dict: 
    """
    df = pd.read_csv(annotated_csv, index_col=0)
    df = df.query('keep==1')
    
    variables = {i:df.loc[df['Source']==i,'Variable'].to_list() for i in df['Source'].unique()}
    
    # add id variables
    variables['id'] = ['visit_occurrence_id','person_id']
    
    return variables

def retain_variables(df, variables):
    """Which variables in the full data frame to keep.
    
    Arguments:
        df (pd.DataFrame): pre-processed dataframe
        variables (dict): keys have source, values are list of variables to keep
        
    Returns:
        pd.DataFrame: pre-processed and slimmed down to only retain variables for model    
    """
    return df.loc[:,[var for k,v in variables.items() for var in v]]


def write_df(df, filename):
    df.to_csv(filename, index=False)
    
def curb65(df):
    """Calculate CURB65 score based on first vital measurement.
    
    1-pt for each, GCS<15, BUN>19, RR≥30, (sBP<90 OR dbP≤60), 
    AGE≥65. Assumes lab value in df is first lab as well. 
    Uses GCS < 15 instead of confusion score (?) in reference.
    
    Returns:
        pd.DataFrame: with value added into col='curb65'
    
    Reference:
        https://www.mdcalc.com/curb-65-score-pneumonia-severity
    """
    df['curb65'] = (df['first_GCS']<=15).astype(int) + (df['bun']>=19).astype(int) + (df['first_RR']>=30).astype(int) + ((df['first_SBP']<90)|(df['first_DBP']<=60)).astype(int) + (df['age_years']>=65)
    return df

def qSOFA(df):
    """Calculate qSOFA score based on worst measurements.
    
    1-pt for each, GCS=<15, RR≥22, sBP<=100.
    
    Returns:
        pd.DataFrame: with value added into col='curb65'
    
    Reference:
        https://www.mdcalc.com/qsofa-quick-sofa-score-sepsis
    """
    df['qSOFA'] = (df['min_GCS']<=15).astype(int) + (df['max_RR']>=22).astype(int) + (df['min_SBP']<=100)
    return df

def dummify(df, verbose=False):
    """Get dummy variables for all vars as object.  
    
    NOTE: does not allow collinearity, i.e., returns `k-1` dummies
    
    Arguments:
        verbose (bool): default=False; if True, checks 
            by printing difference between columns
    
    Returns:
        pd.DataFrame: dummy
    """
    
    temp = pd.get_dummies(df, drop_first=True)
    
    if verbose:
        not_in_orig = [i for i in temp.columns.to_list() if i not in df.columns.to_list()]
        print('Dummified cols (expected?)')
        for i in not_in_orig:
            print('  {}'.format(i))
        
    return temp








if __name__ == '__main__':
    
    
    # pre-processing steps to add to merge_pp()
    ## TODO: get_sparkjosn(), specify file layout
    ## TODO: main() into nice py script
    ## TODO: demographics/demo json file may be corrupted (previously read id as float & with missingness)
    ## TODO: add timings
    # read json files 
    base_path = '/home/jovyan/work/ngr4/ed_data/'
    pdfp = '/home/jovyan/work/ngr4/data/processed/'

    spark = (SparkSession.builder.appName("ED_process").getOrCreate())

    diag = spark.read.json(os.path.join(base_path, 'diagnoses'))
    ed_extra = spark.read.json(os.path.join(base_path, "ed_extra"))
    rads = spark.read.json(os.path.join(base_path, "rad_xray"))
    labs = spark.read.json(os.path.join(base_path, "labs"))
    outcomes = spark.read.json(os.path.join(base_path ,"outcomes"))
    outcomes_header = spark.read.json(os.path.join(base_path, "outcomes_header"))
    # vitals = spark.read.json(os.path.join(base_path, "vitals"))
    demo = spark.read.json(os.path.join(base_path, "demographics"))
    admissions = spark.read.json(os.path.join(base_path, "admissions"))

    admit_df = pp_admissions(admissions)
    # NOTE: change to spark json? if yes, add todo
    demo_df = pp_demographics(load_csv(os.path.join(base_path, os.path.join('demographics','demo.csv')))) # demo_df = pp_demographics(demo) 
    labs_df = pp_labs(labs) # takes awhile, visit_occurence_id is index
    ed_df = pp_ed_extra(load_csv(os.path.join(base_path, os.path.join('ed_extra', 'ed_extra2.csv'))))
    outcomes_df = add_header(outcomes, outcomes_header)
    outcomes_df = pp_outcomes(outcomes_df)
    vitals_df = load_csv(os.path.join(base_path, os.path.join('vitals','vitals.csv')))
    vitals_df = pp_vitals(vitals_df)
    meds_df = pp_meds(load_csv(os.path.join(base_path, os.path.join('meds','ed_outpt_meds.csv'))))
    xray_df = pp_xrays(load_csv(os.path.join(base_path, os.path.join('rad_xray','rads_reduced_adh-20200422.csv'))))
    site_df = pp_site(load_csv(os.path.join(base_path, os.path.join('site','ed_department.csv'))))

    df = merge_pp(outcomes_df, demo_df, 
                  ed_df, labs_df, vitals_df, 
                  meds_df, xray_df, site_df, 
                  check_right_duplicates=True,
                  verbose=True)
    df['primary_outcome_yn'].value_counts()

    if True:
        # clear memory
        del outcomes_df, demo_df, ed_df, labs_df, vitals_df, meds_df, xray_df, site_df

    if False:
        # first time? then set True
        vardf = vars2dict(dfs={'outcomes':outcomes_df, 
                               'demo':demo_df, 
                               'ed':ed_df, 
                               'labs':labs_df, 
                               'vitals':vitals_df, 
                               'meds':meds_df, 
                               'xray':xray_df, 
                               'site':site_df}, 
                           save=os.path.join(base_path,'data_dict.csv'))

    if True:    
        variables = vars2keep(os.path.join(base_path,'data_dict_at.csv'))
    # filter variables
    df = retain_variables(df, variables)
    # add last caclutions 
    df = curb65(df)
    df = qSOFA(df)
    # dummmify, post-check
    df = dummify(df, verbose=False)
    if True:
        # save the df without 
        write_df(df, os.path.join(pdfp,'eddata_200423.csv'))
    