'''
icd10reader.py

Dependencies:
  `pip install hcuppy`

nealgravindra, 200403
'''

import os
import pickle
import pandas as pd
import warnings
import time
from hcuppy.cci import CCIEngine
from hcuppy.elixhauser import ElixhauserEngine


def pkl2df(fname):
    """Unpickle that.

    Args:
      fname (str): full filepath, ext., etc.

    Returns:
      Unpickled object

    """

    with open(fname, 'rb') as f:
        X = pickle.load(f)
        f.close()
    return X

def icd102cci(df, idcol='PAT_ENC_CSN_ID', icd10col='CURRENT_ICD10_LIST', verbose=False):
    """Take df of with a list of ICD10 codes and make wide df per obs.

    Args:
      df (pd.DataFrame): must contain at least a ICD10 column

    Returns:
      Wide df, where columns take body_system_desc.

    Raises:
      Warning if hcuppy has an attribute error in ICD10 codes. Improper entry?

    NOTE: for loop construction is time consuming. Assuming linear
      order, expect ~20s per 10000 pts.

    NOTE2: NaN values are kept when patient doesn't have observation since these
      can easily be converted to 0s later.

    """

    if verbose:
        print('Starting construction of cci df...')
        start = time.time()
    ce = CCIEngine()
    if idcol is not None:
        cci = pd.DataFrame(df[idcol], index=df.index)
    else:
        cci = pd.DataFrame(index=df.index)
    for p in df.index:
        # NOTE: `hcuppy` can't handle nans, throw out
        icd10list = [j for j in df.loc[p,icd10col] if not pd.isna(j)]
        try:
            ce.get_cci(icd10list)
        except AttributeError:
            warnings.warn('\n  `hcuppy` error for obs in df_idx={}\n  obs skipped.'.format(p))
            continue
        for code in ce.get_cci(icd10list):
            if code['is_chronic']:
                cci.loc[p,'chronic{}'.format(code['body_system_desc'].replace(' ','_'))] = 1
            else:
                cci.loc[p,code['body_system_desc'].replace(' ','_')] = 1
    if verbose:
        print('... processed {}-records in {:.2f}-s'.format(df.shape[0],time.time()-start))

    return cci

def icd102eh(df, idcol='PAT_ENC_CSN_ID', icd10col='CURRENT_ICD10_LIST', verbose=False):
    """Take df of with a list of ICD10 codes and make wide df per obs
    with Elixhauser scores and list of comorbidities.

    Args:
      df (pd.DataFrame): must contain at least a ICD10 column

    Returns:
      Wide df, where columns take cmrbdt_list, rdmsn_scr, mrtlt_scr.

    Raises:
      Warning if hcuppy has an attribute error in ICD10 codes. Improper entry?

    NOTE: for loop construction is time consuming. Assuming linear
      order, expect ~15s per 10000 pts.

    NOTE2: NaN values are kept when patient doesn't have observation since these
      can easily be converted to 0s later.

    """

    if verbose:
        print('Starting construction of Elixhauser df...')
        start = time.time()
    ee = ElixhauserEngine()
    if idcol is not None:
        eh = pd.DataFrame(df[idcol], index=df.index)
    else:
        eh = pd.DataFrame(index=df.index)
    for p in df.index:
        # NOTE: `hcuppy` can't handle nans, throw out
        icd10list = [j for j in df.loc[p,icd10col] if not pd.isna(j)]
        try:
            ee.get_elixhauser(icd10list)
        except AttributeError:
            warnings.warn('\n  `hcuppy` error for obs in df_idx={}\n  obs skipped.'.format(p))
            continue
        if isinstance(ee.get_elixhauser(icd10list),dict):
            code = ee.get_elixhauser(icd10list)
            eh.loc[p,'rdmsn_scr'] = code['rdmsn_scr']
            eh.loc[p,'mrtlt_scr'] = code['mrtlt_scr']
            for k in code['cmrbdt_lst']:
                eh.loc[p,'cc_{}'.format(k.lower())] = 1
        else:
            # assume list of dicts
            for code in ee.get_elixhauser(icd10list):
                eh.loc[p,'rdmsn_scr'] = code['rdmsn_scr']
                eh.loc[p,'mrtlt_scr'] = code['mrtlt_scr']
                for k in code['cmrbdt_lst']:
                    eh.loc[p,'cc_{}'.format(k.lower())] = 1
    if verbose:
        print('... processed {}-records in {:.2f}-s'.format(df.shape[0],time.time()-start))

    return eh

def test_cci(fname, verbose=True):
    """Test CCI df construction.

    Args:
      fname (str): full filename w/path to data.

    Returns:
      df and prints head
    """

    df = icd102cci(pkl2df(fname), verbose = True)

    if verbose:
        print(df.head())
    return df

def test_eh(fname, verbose=True):
    """Test Elixhauser df construction.

    Args:
      fname (str): full filename w/path to data.

    Returns:
      df and prints head
    """

    df = icd102eh(pkl2df(fname),verbose=True)

    if verbose:
        print(df.head())
    return df


if __name__ == '__main__':
    dfp = '/home/ngr/ushare/covid_ed/data/'
    fname = 'sample_pmh.pkl'

    cci = test_cci(os.path.join(dfp,fname))
    eh = test_eh(os.path.join(dfp,fname))
