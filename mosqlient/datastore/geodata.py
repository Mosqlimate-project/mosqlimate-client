import geojson
import requests
import warnings
import pandas as pd
from io import StringIO
import geopandas as gpd

URL = "https://info.dengue.mat.br/geoserver/wfs"

def get_data(typeName:str, outputFormat:str):
    '''
    Function that returns a GeoDataFrame or Dataframe stored under the name `{typeName}`  
    in GeoServer.

    Parameters
    -----------
    typeName: str 
        The name of the layer stored in GeoServer.  
    outputFormat: str
        The type of the data saved in the GeoServer. Options: 'json', 'csv'. 

    Returns 
    ---------
    data: gpd.GeoDataFrame or pd.DataFrame
        The requested GeoDataFrame or DataFrame.
    '''

    params = dict(
    service="WFS",
    version="2.0.0",
    request="GetFeature",
    typeName=typeName,
    outputFormat=outputFormat,
)

    r = requests.get(URL, params=params)

    if r.status_code != 200:
        raise RuntimeError(f"Request failed with status code {r.status_code}")
    
    if outputFormat == 'json':
        data = gpd.GeoDataFrame.from_features(geojson.loads(r.content), crs="EPSG:4674")

    if outputFormat == 'csv':
        csv_data = StringIO(r.text)  # Convert response text to file-like object
        data = pd.read_csv(csv_data)
        
    return  data

def get_shape(scale:str = 'muni', state:str = 'all'): 
    """
    Retrieves the shapefile of Brazilian regions as a GeoDataFrame. 

    Parameters
    ----------
    scale : str, default='muni'
        Specifies the level of geographical aggregation. Available options:
        - 'sub_district': Sub-district level .
        - 'district': District level.
        - 'city': Municipality level.
        - 'state': State level.
        - 'regional': Regional Health regions.
        - 'macro_regional': Macro regional health regions.

        Some scales, such as 'state', 'regional', and 'macro_regional', do not support filtering by state.
        So the data for all the country will be returned. 

    state : str, optional (default='all')
        Specifies the state for which data should be returned. 
        - Use a two-letter state abbreviation (e.g., 'SP', 'RJ') to filter by a specific state.
        - Use 'all' to retrieve data for all Brazilian states.

        Filtering by state is only applicable to 'sub_district', 'district', and 'city' scales.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the requested geographical data.
    
    Raises
    ------
    UserWarning
        If `state` is specified for an aggregation level that does not support state filtering.
    
    Notes
    -----
    - The function relies on `get_data` to fetch geospatial data.
    """

    valid_scales = {'sub_district', 'district', 'city', 'state', 'regional', 'macro_regional'}
    valid_states = {'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG',
                    'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO', 'all'}
    
    if state not in valid_states:
        raise ValueError(f"Invalid state: {state}. Use a two-letter state abbreviation or 'all' for all states.")
    
    if scale in {'state', 'regional', 'macro_regional'} and state != 'all':
        warnings.warn(f"The scale = {scale} does not allow filtering by state. Shapes for the entire country will be returned.", category=UserWarning)
    
    if scale == 'sub_district':
        if state == 'all':
            data_frames = [get_data(f'brasil_censo_gpkg:{uf}_subdistritos_CD2022', 'json') for uf in valid_states if uf != 'all']
            data = gpd.GeoDataFrame(pd.concat(data_frames, ignore_index=True))
        else:
            data = get_data(f'brasil_censo_gpkg:{state}_subdistritos_CD2022', 'json')
    
    elif scale == 'district':
        if state == 'all':
            data_frames = [get_data(f'brasil_censo_gpkg:{uf}_distritos_CD2022', 'json') for uf in valid_states if uf != 'all']
            data = gpd.GeoDataFrame(pd.concat(data_frames, ignore_index=True))
        else:
            data = get_data(f'brasil_censo_gpkg:{state}_distritos_CD2022', 'json')
    
    elif scale == 'city':
        if state == 'all':
            data = get_data('brasil_geobr_gpkg:brasi_municipios', 'json')
        else:
            data = get_data(f'brasil_geobr_gpkg:brasil_muni_{state}', 'json')
    
    elif scale == 'state':
        data = get_data('brasil_geobr_gpkg:brasil_ufs', 'json')
    
    elif scale == 'regional':
        data = get_data('brasil_macrorregioes_saude:brasil_regioes_saude', 'json')
        data = data.rename(columns={'cd_rg__': 'code_regional'})
    
    elif scale == 'macro_regional':
        data = get_data('brasil_macrorregioes_saude:brasil_macrorregioes_saude', 'json')
    
    else:
        raise ValueError(f"Invalid scale: {scale}. Valid options are: {', '.join(valid_scales)}")
    
    return data 

def get_map_regional_health_BR(): 
    '''
    It returns a DataFrame that maps each city to its corresponding regional and macro-regional health divisions.

    Returns
    -------
    pd.DataFrame
    '''
   
    data = get_data('brasil_macrorregioes_saude:brasil_macrorregioes_saude_correspondencias', 'csv')

    data = data.drop('populacao_ibge_2022', axis = 1)

    rename_cols = { 'co_regiao_pais':'code_region', 'regiao_pais':'name_region', 
                'co_uf': 'code_state', 'sg_uf': 'abbrev_state',  'uf': 'name_state',
       'cod_macrorregiao_de_saude':'code_macro', 'macrorregiao_de_saude': 'name_macro',
       'cod_regiao_de_saude': 'code_regional', 'regiao_de_saude': 'name_regional',
                'cod_municipio': 'code_muni', 'no_municipio': 'name_muni'}

    data = data.rename(columns = rename_cols)

    return data 