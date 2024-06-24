import json
import numpy as np
import pandas as pd
import altair as alt
from typing import Optional 
from mosqlient import get_prediction_by_id 
from scoringrules import crps_normal, logs_normal
from sklearn.metrics import mean_squared_error, mean_absolute_error


def transform_json_to_dataframe(res:dict) -> pd.DataFrame:
    """
    A function that transforms the prediction output from the API and transforms it in a DataFrame.

    Parameters:
    rest (dict): Output of the  prediction's API.

    Returns:
    pd.DataFrame. 
    """

    json_struct = json.loads(res['prediction'])    
    df = pd.json_normalize(json_struct)
    df.dates = pd.to_datetime(df.dates)

    return df


def evaluate_point_metrics(y_true, y_pred, metric):
    """
    Evaluate multiple sklearn metrics on given true and predicted values.

    Parameters:
    -------------
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    metrics (str): Options: ['MAE', 'MSE'] .

    Returns:
    Scores.
    """

    if metric == 'MAE':

        m = mean_absolute_error

    if metric == 'MSE':
        
        m = mean_squared_error
    
    score = m(y_true, y_pred)
        
    return score

def plot_bar_score(data:pd.DataFrame, score:str) -> alt.Chart:
    '''
    Function to plot a bar chart based on scorer.summary dataframe

    Parameters:
    --------------
    data: pd.DataFrame
    score: str
        Valid options are: ['mae', 'mse', 'crps', 'log_score']
    '''
    data = data.reset_index()

    data['id'] = data['id'].astype(str)   

    bar_chart = alt.Chart(data).mark_bar().encode(

    x=alt.X('id:N', axis=alt.Axis(labelAngle=360)).title('Model'),
    y=alt.Y(f'{score}:Q').title(score),
    color=alt.Color('id', legend=alt.Legend(title='Model'))

    ).properties(
        title=f'{score} score', 
        width = 400, 
        height = 300,
    )

    return bar_chart


def plot_score(data:pd.DataFrame, df_melted:pd.DataFrame, score:str = 'CRPS') -> alt.Chart:
    '''
    Function that returns an Altair panel with the time series of cases and the time series of the score for each model.
    
    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame with the time series of cases must contain the columns `dates` and `casos`.
    df_melted : pd.DataFrame
        The DataFrame must contains the columns: 
        * dates: with the dates';
	    * variable: with the models name;
     	* '{score}_score': with the score value
    '''

    if score == 'CRPS':
        title = 'CRPS score'
        subtitle = 'Lower is better'

    if score == 'log':
        title = 'Log score'
        subtitle = 'Bigger is better'
    
    
    timedata =  alt.Chart(data).mark_line().encode(
    x='dates',
    y='casos',
    color=alt.value('black')).properties(
        width=400,  # Set the width
        height=300  # Set the height
    )

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection_point(nearest=True, on="pointerover",
                                  fields=["dates"], empty=False)
    
    graph_score = alt.Chart(df_melted).mark_point(filled= False).encode(
        x='dates',
        y=f'{score}_score',
        color=alt.Color('variable', legend = alt.Legend(legendX = 100)), 
        ).properties(
        width=400,  # Set the width
        height=250  # Set the height
    )
    
    
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df_melted).mark_point().encode(
        x="dates",
        opacity=alt.value(0),
    ).add_params(
        nearest
    )
    
    # Draw points on the line, and highlight based on selection
    points = graph_score.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw a rule at the location of the selection
    columns = list(df_melted.variable.unique())
    tooltip = [alt.Tooltip(c, type="quantitative", format=".2f") for c in columns]
    tooltip.insert(0,alt.Tooltip('dates:T', title='Date'))
    rules = alt.Chart(df_melted).transform_pivot(
        "variable",
        value=f"{score}_score",
        groupby=["dates"]
    ).mark_rule(color="gray").encode(
        x="dates",
        opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
        tooltip=tooltip,
    ).add_params(nearest)
    
    
    return timedata.properties(
        width=400,  # Set the width
        height=150,  # Set the height
        title='New cases'
    ) & alt.layer(
        graph_score,  points, rules
    ).properties(
        title={
            'text': title,
            'subtitle': subtitle
        }
    )

class Scorer: 
    """
    A class to compare the score of the models.

    Attributes
    ----------

    df_true: pd.DataFrame 
        DataFrame of the cases provided by the user. 

   
    filtered_df_true: pd.DataFrame 
        DataFrame of the cases provided by the user filtered according
        to the interval of the predictions or with the `set_date_range` method . 
        
    ids: Optional[list[int]] 
        The list of the predictions id that will be compared 

    
    dict_df_ids: dict[pd.DataFrame]
        A dict of DataFrames of the predictions. If the key is int it refers to the ids passed in the init. If it is `preds` 
        it refers to the dataframe of the predictions provided by the user. 
    
    filtered_dict_df_ids: dict[pd.DataFrame]
        A dict of DataFrames of the predictions. If the key is int it refers to the ids passed in the init. If it is `preds` 
        it refers to the dataframe of the predictions provided by the user. The DataFrames are filtered according
        to the interval of the predictions or with the `set_date_range` method.
        
    min_date: str
        Min date that will include the information of the df_true and predictions. 

    min_date: str
        Max date that will include the information of the df_true and predictions. 
        
    mae : dict
        Dict where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the values of the dict are the mean absolute error.

    mse: dict
        Dict where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the values of the dict are the mean squared error.

    crps: tuple of dicts
        Dict where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the CRPS score computed for every predicted point, 
        and the second one contains the mean values of the CRPS score for all the points. 

        The CRPS computed assumes a normal distribution. 

    log_score: tuple of dicts
        Dict where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the log score computed for every predicted point, 
        and the second one contains the mean values of the log score for all the points. 

        The log score computed assumes a normal distribution. 


    summary: pd.DataFrame
        DataFrame where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the columns are the scores: mae, mse, and the mean of crps and log_score. 
        
    
    Methods
    -------
    start_date_range():
        Train the model.
    plot_predictions():
        Function that returns an Altair panel (alt.Chart) with the time series of cases and the predictions for each model.
    plot_crps():
        alt.Chart: Method that returns an Altair panel with the time series of cases and the time series 
                    of the CRPS score for each model.
    plot_log_score():
        alt.Chart: Method that returns an Altair panel with the time series of cases and the time series 
                    of the log score for each model.
    plot_mae():
        alt.Chart : Bar chart of the MAE score for each prediction.
    plot_mse():
        alt.Chart : Bar chart of the MSE score for each prediction.
    """
    
    def __init__(self, df_true: pd.DataFrame, ids: Optional[list[int]|list[str]] = None, preds: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        df_true: pd.DataFrame
            DataFrame with the columns `dates` and `casos`. 
        ids : list[int] 
            List of the predictions ids that it will be compared.
        preds: pd.DataFrame 
            Pandas Dataframe already in the format accepted by the platform that will be computed the score.

        """

        # input validation data 
        cols_df_true = ["dates", "casos"]
        
        if not set(cols_df_true ).issubset(set(list(df_true.columns))):
            raise ValueError(
                "Missing required keys in the df_true:"
                f"{set(cols_df_true).difference(set(list(df_true.columns)))}")

        # Ensure all the dates has the same lenght 
        min_dates = [min(df_true.dates)]
        max_dates = [max(df_true.dates)]
        
        dict_df_ids = {}
        
        if preds is not None: 
            cols_preds = ['dates', 'lower', 'preds', 'upper']
            if not set(cols_preds).issubset(set(list(preds.columns))):
                raise ValueError(
                    "Missing required keys in the preds:"
                    f"{set(cols_preds).difference(set(list(preds.columns)))}")

            dict_df_ids['preds'] = preds
            min_dates.append(min(preds.dates))
            max_dates.append(max(preds.dates))
        
        if ( ids is None or len(ids) == 0) and (preds is None): 
            raise ValueError("It must be provide and id or DataFrame to be compared")

        
        if ids is not None: 
            ids = [str(id_) for id_ in ids]
            for id_ in ids: 
                try: 
                    df_ = transform_json_to_dataframe(get_prediction_by_id(id = int(id_)))
                    df_ = df_.sort_values(by = 'dates')
                    dict_df_ids[id_] = df_
                    min_dates.append(min(df_.dates))
                    max_dates.append(max(df_.dates))
                    
                except:
                    raise ValueError(f"Invalid prediction_id provided inside ids:'{id_}'")

            
        min_date = max(min_dates)
        max_date = min(max_dates)

        # updating the dates interval
        df_true = df_true.loc[(df_true.dates >= min_date) & (df_true.dates <= max_date)]
        df_true = df_true.sort_values(by = 'dates')
        df_true.reset_index(drop = True, inplace = True)

        for id_ in dict_df_ids.keys(): 
            df_id = dict_df_ids[id_]
            df_id = df_id.loc[(df_id.dates >= min_date) & (df_id.dates <= max_date)]
            df_id = df_id.sort_values(by = 'dates')
            dict_df_ids[id_] = df_id    

        self.df_true = df_true 
        self.filtered_df_true =  df_true
        self.ids = ids 
        self.dict_df_ids = dict_df_ids
        self.filtered_dict_df_ids = dict_df_ids
        self.min_date = min_date
        self.max_date = max_date

    def set_date_range(self, start_date:str, end_date:str)-> None:
        '''
         This method will redefine the interval of dates used to compute the scores.
         The new dates provided must be in the interval defined by the `__init__` method that ensures 
         the df_true and predictions are in the same interval. 
         You can access these values by score.min_date and score.max_date. 

        Parameters
        --------------
        start_date: str
            The new start date used to compute the scores.
        end_date: str
            The new end date used to compute the scores. 
        '''

        if (self.min_date > pd.to_datetime(start_date)) or (self.max_date < pd.to_datetime(start_date)):
            raise ValueError(
                f"The start and end date must be between {self.min_date} and {self.max_date}.")

        df_true = self.df_true
        dict_df_ids = self.dict_df_ids 

        self.filtered_df_true = df_true.loc[(df_true.dates >= pd.to_datetime(start_date)) & (df_true.dates <= pd.to_datetime(end_date))] 
        
        for id_ in dict_df_ids.keys(): 
            df_id = dict_df_ids[id_]
            df_id = df_id.loc[(df_id.dates >= pd.to_datetime(start_date)) & (df_id.dates <= pd.to_datetime(end_date))]
            dict_df_ids[id_] = df_id    

        self.filtered_dict_df_ids = dict_df_ids
        
        return None 

    @property
    def mae(self,):
        '''
        dict: Dict, where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the values of the dict are the mean absolute error.
        '''
        ids  = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true

        scores = {} 
        
        for id_ in dict_df_ids.keys():

            scores[id_] =  evaluate_point_metrics(df_true.casos, y_pred = dict_df_ids[id_].preds,
                                                  metric = 'MAE')

        return scores

    @property
    def mse(self,):
        '''
        dict: Dict, where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the values of the dict are the mean squared error.
        '''

        ids  = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true
 
        scores = {} 
        
        for id_ in dict_df_ids.keys():

            scores[id_] =  evaluate_point_metrics(df_true.casos, y_pred = dict_df_ids[id_].preds,
                                                  metric = 'MSE')

        return scores

    @property
    def crps(self,):
        '''
        tuple of dict: Dict where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the CRPS score computed for every predicted point, 
        and the second one contains the mean values of the CRPS score for all the points. 

        The CRPS computed assumes a normal distribution.
        '''

        ids  = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true

        scores_curve = {} 

        scores_mean = {}
        
        for id_ in dict_df_ids.keys():

            df_id_ = dict_df_ids[id_]

            score = crps_normal(df_true.casos, df_id_.preds, (df_id_.upper-df_id_.lower)/4)

            scores_curve[id_] = pd.Series(score, index=df_true.dates)

            scores_mean[id_] =  np.mean(score)

        self.crps_curve = scores_curve
        
        return scores_curve, scores_mean

    @property
    def log_score(self,):
        '''
        tuple of dict: Dict where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the log score computed for every predicted point, 
        and the second one contains the mean values of the log score for all the points. 

        The log score computed assumes a normal distribution. 
        '''
        
        ids  = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true

        scores_curve = {} 
        scores_mean = {}
        
        for id_ in dict_df_ids.keys():

            df_id_ = dict_df_ids[id_]
            score = logs_normal(df_true.casos, df_id_.preds, (df_id_.upper-df_id_.lower)/4, negative = False)
            scores_curve[id_] = pd.Series(score, index=df_true.dates)
            scores_mean[id_] =  np.mean(score)

        self.log_curve = scores_curve
        
        return scores_curve, scores_mean


    @property 
    def summary(self,):
        '''
        pd.DataFrame: DataFrame where the keys are the id of the models or `preds` when a dataframe of predictions is provided by the user,
        and the columns are the scores: mae, mse, and the mean of crps and log_score. 
        '''
        sum_scores = {}

        sum_scores['mae'] = self.mae
        
        sum_scores['mse'] = self.mse
        
        sum_scores['crps'] = self.crps[1]
        
        sum_scores['log_score'] = self.log_score[1]

        df_score = pd.DataFrame.from_dict(sum_scores, orient='columns')

        df_score.index.name = 'id'

        return df_score 
    
    def plot_mae(self,) -> alt.Chart:
        '''
        Bar chart of the MAE score for each prediction.
        '''

        return plot_bar_score(self.summary, 'mae')
    

    def plot_mse(self,) -> alt.Chart:
        '''
        Bar chart of the MSE score for each prediction.
        '''

        return plot_bar_score(self.summary, 'mse')


    def plot_crps(self,) -> alt.Chart:
        '''
        alt.Chart: Function that returns an Altair panel with the time series of cases and the time series of the CRPS score for each model
        '''
        

        crps_ = self.crps_curve

        df_crps = pd.DataFrame()

        for v in crps_.keys():

            df_crps[str(v)] = crps_[v]
            
        df_crps.reset_index(inplace = True)

        df_melted = pd.melt(df_crps,id_vars = 'dates', value_vars=list(map(str, crps_.keys())))
        df_melted = df_melted.rename(columns = {'value':'CRPS_score'})

        return plot_score(self.df_true, df_melted, score = 'CRPS')


    def plot_log_score(self,) -> alt.Chart:
        '''
        alt.Chart: Function that returns an Altair panel with the time series of cases and the time series of the Log score for each model
        '''

        crps_ = self.log_curve

        df_crps = pd.DataFrame()

        for v in crps_.keys():

            df_crps[str(v)] = crps_[v]
            
        df_crps.reset_index(inplace = True)

        df_melted = pd.melt(df_crps,id_vars = 'dates', value_vars=list(map(str, crps_.keys())))
        df_melted = df_melted.rename(columns = {'value':'log_score'})

        return plot_score(self.df_true, df_melted, score = 'log')


    def plot_predictions(self,show_ci:bool = True,  width:int = 400, height:int = 300) -> alt.Chart:
        '''
        Function that returns an Altair panel (alt.Chart) with the time series of cases and the predictions for each model
   
        Parameters
        ---------------
        show_ci :bool 
            If True it shows the confidence interval.
        width: int
            width of the plot 
        width: int
            height of the plot 
        '''
        

        dict_df_ids = self.filtered_dict_df_ids
        df_true_ = self.filtered_df_true
        df_true_.loc[:, 'legend'] = 'Data'

        if show_ci:  
            title = "Median and 95% confidence interval"
        else:
            title = 'Median of predictions'
        
        df_to_plot = pd.DataFrame()
            
        for id_ in dict_df_ids.keys():
            
            df_ = dict_df_ids[id_]
            
            df_.loc[:, 'model'] = id_
            
            df_to_plot = pd.concat([df_to_plot, df_])

        df_to_plot['model'] = df_to_plot['model'].astype(str)

        data = alt.Chart(df_true_).mark_circle(size = 60).encode(
            x='dates:T',
            y='casos:Q',
            color=alt.Color('legend:N', scale=alt.Scale(range=['black']), legend=alt.Legend(title=None))
        ).properties(
            width=width,  # Set the width
            height=height  # Set the height
        )
        
        # here we define the plot of the right figure
        timeseries = alt.Chart(df_to_plot, title=title).mark_line(
            ).encode(
            x=alt.X('dates:T').title('Dates'),
            y=alt.Y('preds:Q').title('New cases'),
            color=alt.Color('model', legend=alt.Legend(title='Model'))
        )
        
        # here we create the area that represent the confidence interval of the predicitions
        timeseries_conf = timeseries.mark_area( 
            opacity=0.25,
        ).encode(
            x='dates:T',
            y='lower:Q',
            y2='upper:Q',
            color=alt.Color('model', legend=None)
        )

        nearest = alt.selection_point(nearest=True, on="pointerover",
                                  fields=["dates"], empty=False)

         # Draw points on the line, and highlight based on selection
        points = timeseries.mark_point().encode(
            color=alt.Color('model', legend=None ),
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            )
        
        df_true_ = df_true_.rename(columns = {'casos': 'preds'} )

        df_true_['model'] = 'cases'
        
        df_to_plot = pd.concat([df_to_plot, df_true_])

        columns = list(df_to_plot.model.unique())
        tooltip = [alt.Tooltip(c, type="quantitative", format=".0f") for c in columns]
        tooltip.insert(0,alt.Tooltip('dates:T', title='Date'))

        rules = alt.Chart(df_to_plot).transform_pivot(
                "model",
                value="preds",
                groupby=["dates"]
            ).mark_rule(color="gray").encode(
                x="dates",
                opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
                tooltip=tooltip,
            ).add_params(nearest)

        if show_ci:

            final = (data + timeseries + timeseries_conf + points + rules).resolve_scale(
            color='independent'
            )

        else: 
            final = alt.layer(data, 
                timeseries, points, rules
            ).resolve_scale(
            color='independent'
             )

        return final
        

    