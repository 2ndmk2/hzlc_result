"""
Modules to get gaia information of particular stars

Author: masa, naokawa
"""

from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def GetGAIAData(GaiaDR3SourceIDs, FolderForLocalStorage):
    # gets the GAIA data for the provided GaiaDR2SourceIDs's
    # and writes to a local CSV
        
    dfGaia = pd.DataFrame()
    
    #job = Gaia.launch_job_async( "select top 100 * from gaiadr2.gaia_source where parallax>0 and parallax_over_error>3. ") # Select `good' parallaxes
    qry = "SELECT * FROM gaiadr3.gaia_source gs WHERE gs.source_id in (" + GaiaDR3SourceIDs + ");"
    
    job = Gaia.launch_job_async( qry )
    tblGaia = job.get_results()       #Astropy table
    dfGaia = tblGaia.to_pandas()      #convert to Pandas dataframe
    
    npGAIARecords = dfGaia.to_numpy() #convert to numpy array    
    lstGAIARecords = [list(x) for x in npGAIARecords]   #convert to List[]
    
    FileForLocalStorage = FolderForLocalStorage + str(lstGAIARecords[0][2]) + '.csv'  # use SourceID from 1st record
    dfGaia.to_csv (FileForLocalStorage, index = False, header=True) 
    
    column = ["source_id", "phot_g_mean_mag", "bp_rp", "parallax", "phot_variable_flag", "teff_gspphot"]
    column_new = ["source_id", "gabs", "bp_rp", "phot_g_mean_mag",  "parallax", "phot_variable_flag", "teff_gspphot", ]
    dfGaia["gabs"] = dfGaia["phot_g_mean_mag"] + 5 * np.log10(dfGaia["parallax"])-10.0
    df_out = dfGaia[column_new ]    
    
    ReductedFileForLocalStorage = FolderForLocalStorage + 'Reducted' + str(lstGAIARecords[0][2]) + '.csv'
    df_out.to_csv (ReductedFileForLocalStorage, index = False, header=True) 
    
    return df_out

def make_gaiaid_str(gaia_ids):
    str_now = ""
    for (i, gaia_id) in enumerate(gaia_ids):
        if i==len(gaia_ids)-1:
            str_now += str(gaia_id) 
        else:
            str_now += str(gaia_id) + ", "

    return str_now

def make_df_ref(path_to_ref):
    with open(path_to_ref) as f:
        columns = f.readlines()[1]
    columns = columns.strip('# columns:')
    columns_list = columns.split(',')
    for i in range(len(columns_list)):
        columns_list[i] = columns_list[i].strip()
        
    df_ref = pd.read_csv(path_to_ref, sep='|', comment='#', header=None)
    
    for i in range(len(columns_list)):
        df_ref = df_ref.rename(columns={i:columns_list[i]})
        
    print(df_ref)
    
    df_ref["gabs"] = df_ref["phot_g_mean_mag"] + 5 * np.log10(df_ref["parallax"])-10.0
    
    return df_ref
    
def plot_HR(df,df_ref):
    
    gabs = df["gabs"].values
    c = df["bp_rp"].values
    
    gabs_ref = df_ref["gabs"].values
    bp_mag_ref = df_ref['phot_bp_mean_mag'].values
    rp_mag_ref = df_ref['phot_rp_mean_mag'].values
    c_ref = bp_mag_ref - rp_mag_ref
    
    fig = plt.figure(figsize=(10,10))
    
    x = c
    y = gabs
    
    x_ref = c_ref
    y_ref = gabs_ref
    
    plt.scatter(x,y,c='red', zorder=1, s=50)
    plt.scatter(x_ref,y_ref, zorder=0, c=x_ref, s=0.6, cmap ='jet', alpha=0.3)
    
    
    plt.ylim(15,-20)
    plt.xlabel('bp_rp')
    plt.ylabel('gabs')

    plt.grid()
    plt.show()