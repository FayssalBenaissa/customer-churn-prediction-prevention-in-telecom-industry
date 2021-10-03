import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Row import Row
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash import no_update
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import os
import sys
import copy
import time
from src.graphs import df, layout, ohe, cat_features, svm_model, voting_model
from content import tab_prediction_content, tab_analysis_content ,Show_content,tab_Survival,tab_temps_survie,tab_covariate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve 
from sklearn.metrics import brier_score_loss
import lifelines
import flask
import glob
import os
# importing the data 
df= pd.read_csv("Telco-Customer-Churn.csv")
# Creating the app

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets = [dbc.themes.SUPERHERO,'/assets/styles.css']
) 

server=app.server

# Tabs Content

tabs = dbc.Tabs(
    [
        dbc.Tab(tab_prediction_content, label="Prediction"),
        dbc.Tab(tab_analysis_content, label="Data Analysis"),
        dbc.Tab(Show_content, label="Show Data"),
        dbc.Tab(tab_Survival,label="Survival analysis"),
    ]
)


jumbotron = dbc.Jumbotron(
    [
        # html.H1("Jumbotron", className="display-3"),
        # html.P(
        #     "Use a jumbotron to call attention to "
        #     "featured content or information.",
        #     className="lead",
        # ),
        # html.Hr(className="my-2"),
        html.H4("Customer Churn Analysis and Prediction with ML and Survival Analysis"),
        # html.P(dbc.Button("Learn more", color="primary"), className="lead"),
    ], className="cover"
)


# App Layout

app.layout = html.Div(
    [
                
        jumbotron,
        html.Div(
            [
                
                dbc.Row(dbc.Col(tabs, width=12)),
                
                
            ], id="mainContainer",style={"display": "flex", "flex-direction": "column"}
        ),
        

        #html.P("Developed by Tolgahan Çepel", className="footer")


    ],
)

# Callbacks

@app.callback(
    Output("categorical_bar_graph", "figure"),
    [
        Input("categorical_dropdown", "value"),
    ],
)

def bar_categorical(feature):

    time.sleep(0.2)

    temp = df.groupby([feature, 'Churn']).count()['customerID'].reset_index()
    
    fig = px.bar(temp, x=feature, y="customerID",
             color=temp['Churn'].map({'Yes': 'Churn', 'No': 'NoChurn'}),
             color_discrete_map={"Churn": "#47acb1", "NoChurn": "#f26522"},
             barmode='group')
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    _title = (feature[0].upper() + feature[1:]) + " Distribution by Churn"
    
    fig.update_layout(
        title = {'text': _title, 'x': 0.5},
        #xaxis_visible=False,
        xaxis_title="",
        yaxis_title="Count",
        legend_title_text="",
        legend = {'x': 0.16}
    )
    return fig

@app.callback(
    Output("categorical_pie_graph", "figure"),
    [
        Input("categorical_dropdown", "value"),
    ],
)

def donut_categorical(feature):

    time.sleep(0.2)

    temp = df.groupby([feature]).count()['customerID'].reset_index()

    fig = px.pie(temp, values="customerID", names=feature, hole=.5,
                            #color=temp['Churn'].map({'Yes': 'Churn', 'No': 'NoChurn'}),
                                #color_discrete_map={"Churn": "#47acb1",
                                                            #"NoChurn": "#f26522"},
    )

    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    _title = (feature[0].upper() + feature[1:]) + " Percentage"

    if(df[feature].nunique() == 2):
        _x = 0.3
    elif(df[feature].nunique() == 3):
        _x = 0.16
    else:
        _x = 0

    fig.update_layout(
        title = {'text': _title, 'x': 0.5},
        legend = {'x': _x}
    )



    return fig


# Prediction

@app.callback(
    [dash.dependencies.Output('vote_result', 'children')],

    [dash.dependencies.Input('btn_predict', 'n_clicks')],

    [dash.dependencies.State('ft_gender', 'value'),
     dash.dependencies.State('ft_partner', 'value'),
     dash.dependencies.State('ft_dependents', 'value'),
     dash.dependencies.State('ft_phoneService', 'value'),
     dash.dependencies.State('ft_multipleLines', 'value'),
     dash.dependencies.State('ft_internetService', 'value'),
     dash.dependencies.State('ft_onlineSecurity', 'value'),
     dash.dependencies.State('ft_onlineBackup', 'value'),
     dash.dependencies.State('ft_deviceProtection', 'value'),
     dash.dependencies.State('ft_techSupport', 'value'),
     dash.dependencies.State('ft_streamingTv', 'value'),
     dash.dependencies.State('ft_streamingMovies', 'value'),
     dash.dependencies.State('ft_contract', 'value'),
     dash.dependencies.State('ft_paperlessBilling', 'value'),
     dash.dependencies.State('ft_paymentMethod', 'value'),
     dash.dependencies.State('ft_seniorCitizen', 'value'),
     dash.dependencies.State('ft_monthlyCharges', 'value'),
     dash.dependencies.State('ft_totalCharges', 'value'),
     dash.dependencies.State('ft_tenure', 'value')]
)

def predict_churn(n_clicks, ft_gender, ft_partner, ft_dependents, ft_phoneService, ft_multipleLines,
                            ft_internetService, ft_onlineSecurity, ft_onlineBackup, ft_deviceProtection,
                            ft_techSupport, ft_streamingTv, ft_streamingMovies, ft_contract,
                            ft_paperlessBilling, ft_paymentMehod, ft_seniorCitizen, ft_monthlyCharges,
                            ft_totalCharges, ft_tenure):

    time.sleep(0.4)

    sample = {'gender': ft_gender, 'Partner': ft_partner, 'Dependents': ft_dependents,
              'PhoneService': ft_phoneService, 'MultipleLines': ft_multipleLines,
              'InternetService': ft_internetService, 'OnlineSecurity': ft_onlineSecurity, 'OnlineBackup': ft_onlineBackup,
              'DeviceProtection': ft_deviceProtection, 'TechSupport': ft_techSupport, 'StreamingTV': ft_streamingTv,
              'StreamingMovies': ft_streamingMovies, 'Contract': ft_contract, 'PaperlessBilling': ft_paperlessBilling,
              'PaymentMethod': ft_paymentMehod, 'TotalCharges': float(ft_totalCharges), 'MonthlyCharges': float(ft_monthlyCharges),
              'tenure': int(ft_tenure), 'SeniorCitizen': int(ft_seniorCitizen)}

    sample_df = pd.DataFrame(sample, index=[0])
    sample_df_enc = ohe.transform(sample_df[cat_features])
    sample_df_enc = pd.DataFrame(sample_df_enc)

    sample_df_enc = pd.concat([sample_df_enc, sample_df[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure']]], axis=1)

    #svm_prediction = svm_model.predict(sample_df_enc)
    voting_prediction = voting_model.predict(sample_df_enc)

    def churn_to_text(num):
        if(num == 0):
            return "Predicted: Not Churn"
        elif(num == 1):
            return "Predicted: Churn"

    # print(svm_prediction)

    if(n_clicks):
        return [churn_to_text(voting_prediction)]
    else:
        return no_update

@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)

# we use a callback to toggle the collapse on small screens
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


#Analysis survival
card_categorical = html.Div(
    [
         dbc.Spinner(size="md",color="light",
                    children=[
                        dcc.Graph(id="categorical_bar_graph1", config = {"displayModeBar": False}, style = {"height": "54vh","padding-left":"30%","padding-right":"30%"},)
                    ]
                ),
                
            
        
    ],
    style = {"background-color": "#16103a",}
)
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])

def render_page_content(pathname):
    if pathname == "/":
       return html.Div(

        tab_temps_survie

)  
    elif pathname == "/page-1":
        return html.Div(
            tab_covariate
        )
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

@app.callback(
    [dash.dependencies.Output('NUm_result', 'children'),
     dash.dependencies.Output('surv_fnc', 'figure'),
     dash.dependencies.Output('rev_result', 'children')
     ],

    [dash.dependencies.Input('btn_predict1', 'n_clicks')],

    [dash.dependencies.State('ft_gender1', 'value'),
     dash.dependencies.State('ft_partner1', 'value'),
     dash.dependencies.State('ft_dependents1', 'value'),
     dash.dependencies.State('ft_phoneService1', 'value'),
     dash.dependencies.State('ft_multipleLines1', 'value'),
     dash.dependencies.State('ft_internetService1', 'value'),
     dash.dependencies.State('ft_onlineSecurity1', 'value'),
     dash.dependencies.State('ft_onlineBackup1', 'value'),
     dash.dependencies.State('ft_deviceProtection1', 'value'),
     dash.dependencies.State('ft_techSupport1', 'value'),
     dash.dependencies.State('ft_streamingTv1', 'value'),
     dash.dependencies.State('ft_streamingMovies1', 'value'),
     dash.dependencies.State('ft_contract1', 'value'),
     dash.dependencies.State('ft_paperlessBilling1', 'value'),
     dash.dependencies.State('ft_paymentMethod1', 'value'),
     dash.dependencies.State('ft_seniorCitizen1', 'value'),
     dash.dependencies.State('ft_monthlyCharges1', 'value'),
     dash.dependencies.State('ft_totalCharges1', 'value'),
     dash.dependencies.State('ft_tenure1', 'value')]
)
def predict_churn(n_clicks, ft_gender1, ft_partner1, ft_dependents1, ft_phoneService1, ft_multipleLines1,
                            ft_internetService1, ft_onlineSecurity1, ft_onlineBackup1, ft_deviceProtection1,
                            ft_techSupport1, ft_streamingTv1, ft_streamingMovies1, ft_contract1,
                            ft_paperlessBilling1, ft_paymentMehod1, ft_seniorCitizen1, ft_monthlyCharges1,
                            ft_totalCharges1, ft_tenure1):

    time.sleep(0.4)
    sample = {'customerID' : 'Ax2021' ,'gender': ft_gender1 , 'SeniorCitizen': int(ft_seniorCitizen1)
    , 'Dependents': ft_dependents1,'tenure': int(ft_tenure1),'PhoneService': ft_phoneService1,'MultipleLines': ft_multipleLines1,
    'InternetService': ft_internetService1,'OnlineSecurity': ft_onlineSecurity1,'OnlineBackup': ft_onlineBackup1,
    'DeviceProtection': ft_deviceProtection1, 'TechSupport': ft_techSupport1, 'StreamingTV': ft_streamingTv1,
   'StreamingMovies': ft_streamingMovies1,'Contract': ft_contract1,'PaperlessBilling': ft_paperlessBilling1,
   'PaymentMethod': ft_paymentMehod1,'TotalCharges': float(ft_totalCharges1),'MonthlyCharges': float(ft_monthlyCharges1),
    'Churn': 'No'}

 
    sample_df = pd.DataFrame(sample,index = [0])
    df2= pd.read_csv("Telco-Customer-Churn.csv")
    df2.iloc[df2[df2['TotalCharges'] == ' '].index, 19] = df2['MonthlyCharges']
    data = pd.concat([df2,sample_df], axis= 0)
    # data = df2.copy()
    # Replace single white space with MonthlyCharges and convert to numeric
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
    data.drop(columns = ['customerID'], inplace=True)

    # Convert Churn column to 1 (Yes) or 0 (No)
    data['Churn'] = data['Churn'].replace({"No": 0, "Yes": 1})

    # Create a list of features where we will assign 1 to a Yes value and 0 otherwise
    features_to_combine = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
					   'TechSupport', 'StreamingTV', 'StreamingMovies']
    # Assign 1 to Yes and 0 otherwise
    for feat in features_to_combine:
        data[feat] = data[feat].apply(lambda x: 1 if x == 'Yes' else 0)

    # Create dummy variables for our remaining categorical columns
    data = pd.get_dummies(data, columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
				       'InternetService', 'Contract', 'PaperlessBilling',
				       'PaymentMethod'], drop_first = False)

    # Drop that dummy variable that the business considers to be typical of their subscribers
    data.drop(columns = ['gender_Male', 'Partner_Yes', 'Dependents_No', 'PhoneService_Yes',
		     'InternetService_Fiber optic', 'Contract_Month-to-month', 'PaperlessBilling_Yes',
		     'PaymentMethod_Electronic check'], inplace = True)
   
    cph = lifelines.CoxPHFitter(penalizer = 0.1)
    cph.fit(data, duration_col = 'tenure', event_col = 'Churn'  )
    censored_data = data[data['Churn'] == 0]

    d = data.tail(1)
    survFun = cph.predict_survival_function(d)
    dd = survFun.to_dict()
    SurvT = lifelines.utils.median_survival_times(survFun)
    p = go.Figure(data=[ go.Scatter(x = list(dd[0].keys()) , y = list(dd[0].values() ))])
    revenu=float(SurvT)* float(ft_monthlyCharges1)
    if(n_clicks):
        return [str(SurvT) +" mois", p,str(revenu)]
    else:
        return no_update


image_directory = 'C:/Users/Asus/Desktop/telco-customer-churn-master/telco-customer-churn-master/static/'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('categorical_dropdown1', 'value')])
def update_image_src(value):
    return static_image_route + value

# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)
if __name__ == "__main__":
    app.run_server(debug= 0, port=8050)
