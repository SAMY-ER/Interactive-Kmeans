# *********************************************************************************************************************
#                                   K-MEANS CLUSTERING : INTERACTIVE VISUALIZATION
# *********************************************************************************************************************


# ***************************************
#                IMPORTS
# ***************************************
import os
import json
import numpy as np
from random import shuffle
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from HelpFunctions import dist, create_dataset, init_centroids, Kmeans_EStep, Kmeans_MStep, make_kmeans_viz_data

# ***************************************
#              INITIALIZATION
# ***************************************
app = dash.Dash(__name__)

# ***************************************
#              APPLICATION
# ***************************************

## COLOR PALETTES
clusColorMap = {0:'#6395EC', 1:'#FFA27A', 2:'#DDA618', 3:'#DFA5DD', 4:'#61CCAB', 5:'#CD5C5C', 6:'#719D2F', 7:'#8E4D2C', 8:'#472F63'}
lineColors = {0: 'rgba(99,149,236, .7)', 1: 'rgba(255,162,122, .7)', 2: 'rgba(221,166,24, .7)', 3: 'rgba(223,165,221, .7)', 
              4: 'rgba(97,204,171, .7)', 5: 'rgba(205,92,92, .7)', 6: 'rgba(113,157,47, .7)', 7:'rgba(142,77,44, .7)', 
              8: 'rgba(71,47,99, .7)'}

## BUILD DASH APPLICATION
    ## APP LAYOUT
app.layout = html.Div(children=[

    # COLUMN 1 (TITLE + PARAMETERS BLOCK + K-MEANS GRAPH)
    html.Div([
        # ROW 1 (TITLE)
        html.Div([
            html.H1('K-means : Interactive Visualization', style = {'color' : '#4B4D55', 'font-family':'savoye LET', 'fontSize' : 80, 'fontWeight' : 100, 'textAlign' : 'center', 'marginTop':'2%', 'marginBottom':'0%'})
        ], style = {'height' : '12.5%', 'border-radius':10, 'borderStyle' : 'solid', 'borderWidth' : 2, 'borderColor' : 'black', 'marginRight':'2%', 'marginBottom':'2%', 'background-color':'white'}
        ),
        # ROW 2 (PARAMETERS+GRAPH)
        html.Div([
            # ROW 2 FIRST PART (PARAMETERS)
            html.Div([
                html.Div([
                    html.H6('CREATE DATASET', style = {'text-align':'center', 'marginTop' : 15, 'marginBottom':15, 'fontSize':15, 'fontWeight':800, 'color' : '#4B4D55'}),

                    html.Div([
                        html.Div('Shape :', style={'marginLeft':2}),
                        dcc.Dropdown(
                            id='shape-dropdown',
                            options=[
                                {'label': 'Gaussian Mixture', 'value': 'gmm'},
                                {'label': 'Circles', 'value': 'circle'},
                                {'label': 'Moons', 'value': 'moon'},
                                {'label': 'Anisotropicly distributed', 'value': 'anisotropic'},
                                {'label': 'No Structure', 'value': 'noStructure'},
                            ],
                            value='gmm'
                        )
                    ],  style = {'marginLeft': 10, 'marginRight': 10, 'marginBottom':15}),

                    html.Div([
                        html.Div('Sample Size :', style={'marginLeft':2}),
                        dcc.Slider(id='sample-slider', min=50, max=500, step=50, value=200, marks={50:{'label':50}, 500:{'label':500}})
                    ], style = {'marginLeft': 10, 'marginRight': 10, 'marginBottom':25}),

                    html.Div([
                        html.Div('Nb of Clusters :', style={'marginLeft':2}),
                        dcc.Slider(id='cluster-slider', min=2, max=9, marks={i: '{}'.format(i) for i in range(2, 10)}, value=3)
                    ], style = {'marginLeft': 10, 'marginRight': 10}),

                    html.Div([
                        html.Button('GENERATE DATA', id='regenData-button', style={'width':130, 'font-size':10, 'text-align':'center', 'padding': 0}),
                    ], style={'marginLeft':'22%', 'marginTop':40}),

                ]),

                html.Div([
                    html.H6('INITIALIZE K-MEANS', style = {'text-align':'center', 'marginTop' : 45, 'marginBottom':15, 'fontSize':15, 'fontWeight':800, 'color' : '#4B4D55'}),

                    html.Div([
                        html.Div('Initialization Method :', style={'marginLeft':2}),
                        dcc.Dropdown(
                            id='init-dropdown',
                            options=[
                                {'label': 'Random', 'value': 'random'},
                                {'label': 'K-means++', 'value': 'kmeans++'}
                            ],
                            value='random'
                        )
                    ],  style = {'marginLeft': 10, 'marginRight': 10, 'marginBottom':15}),

                    html.Div([
                            html.Div('Nb of Centroids :', style={'marginLeft':2}),
                            dcc.Slider(id='centroid-slider', min=2, max=9, marks={i: '{}'.format(i) for i in range(2, 10)}, value=3)
                        ], style = {'marginLeft': 10, 'marginRight': 10, 'marginBottom':30}),

                    html.Div([
                            html.Div('Max Iterations :', style={'marginLeft':2}),
                            dcc.Slider(id='iter-slider', min=5, max=20, step=1, value=10, marks={5:{'label':5}, 20:{'label':20}})
                        ], style = {'marginLeft': 10, 'marginRight': 10, 'marginBottom':15}),  

                    html.Div([
                        html.Button('GENERATE CENTROIDS', id='regenCentroids-button', style={'width':130, 'font-size':8, 'text-align':'center', 'padding': 0}),
                    ], style={'marginLeft':'22%', 'marginTop':30}),
                ]),
            ], className='three columns', style = {'background-color':'white', 'height' : '670', 'borderStyle' : 'solid', 'border-radius':10, 'borderWidth' : 2, 'borderColor' : 'black', 'marginRight':'0%'}),
            # ROW 2 SECOND PART (GRAPH)
            html.Div([
                html.H6('K-MEANS SCATTER PLOT', style = {'text-align':'center', 'marginTop' : 30, 'marginBottom':5, 'fontSize':15, 'fontWeight':800, 'color' : '#4B4D55'}),

                html.Button('Play', id='play-button', style={'marginLeft':45, 'marginTop':30, 'marginBottom':10, 'marginRight':5, 'width':60, 'font-size':10, 'text-align': 'center', 'padding': 0}),
                html.Button('Pause', id='pause-button', style={'marginRight':5, 'width':70, 'font-size':10, 'text-align':'center', 'padding': 0}),
                html.Button('<<', id='prevStep-button', style={'marginRight':5, 'width':40, 'font-size':10, 'text-align':'center', 'padding': 0}),
                html.Button('>>', id='nextStep-button', style={'marginRight':5, 'width':40, 'font-size':10, 'text-align':'center', 'padding': 0}),
                html.Button('Restart', id='restart-button', style={'width':90, 'font-size':10, 'text-align':'center', 'padding': 0}),

                html.Button(id='iter-text', disabled=True, style={'marginLeft':95, 'width':180, 'background-color':'#4B4D55', 'pointer-events': 'none', 'color':'white', 'font-size':10, 'text-align':'center', 'padding': 0}),

                dcc.Graph(id='kmeans-graph', animate=True, config={'displayModeBar': False}, style={'marginLeft':5, 'marginRight':15}),
                dcc.Interval(id='interval-component', interval=3600*1000, n_intervals=0),
                html.Img(src=app.get_asset_url('Signature_Logo.png'), style={'width':70, 'bottom':25, 'left':'92%', 'position':'relative'})#775

            ], className = 'nine columns', style = {'background-color':'white', 'height' : '670', 'border-width':2, 'borderStyle' : 'solid', 'border-radius':10, 'borderColor' : 'black', 'marginLeft':'2%'}),
 
        ])
        
    ], className = 'eight columns', style = { 'marginLeft':30, 'marginRight':0}),
    # COLUMN 2 (COST FUNCTION GRAPH + SILHOUETTE GRAPH)
    html.Div([
        html.Div([
            html.H6('K-MEANS COST FUNCTION', style = {'text-align':'center', 'marginTop' : 15, 'marginBottom':5, 'fontSize':15, 'fontWeight':800, 'color' : '#4B4D55'}),
            dcc.Graph(id='inertia-graph', config={'displayModeBar': False}, style={'marginLeft':5, 'marginRight':5}),
        ], style = {'background-color':'white', 'height' : 390, 'marginBottom':25, 'border-radius':10, 'borderStyle' : 'solid', 'borderWidth' : 2, 'borderColor' : 'black'}),
        
        html.Div([
            html.H6('SILHOUETTE ANALYSIS', style = {'text-align':'center', 'marginTop' : 15, 'marginBottom':5, 'fontSize':15, 'fontWeight':800, 'color' : '#4B4D55'}),
            dcc.Graph(id='silhouette-graph', config={'displayModeBar': False}, style={'marginLeft':5, 'marginRight':5})
        ], style = {'background-color':'white', 'height' : 390, 'border-radius':10, 'borderStyle' : 'solid', 'borderWidth' : 2, 'borderColor' : 'black'}),
    ], className = 'four columns', style={ 'height':800, 'marginLeft':0, 'marginRight':0}),

    # COPYRIGHT
    html.Div([
        html.H6('Copyright © 2019  Samy TAFASCA', style={'color':'white', 'font-size':10, 'text-align':'right'}),

        html.Div(id='dataset-value', style={'display': 'none'}),
        html.Div(id='kmeansCentroids-value', style={'display': 'none'}),
        html.Div(id='kmeansFrames-counter', style={'display': 'none'}),
    ], className= 'twelve columns')
]) 


                            # *************************************** #
                            #               CALLBACKS                 #
                            # *************************************** #

# *********************************************************************************************************
#                           CALLBACK 1 : UPDATE DATASET AND STORE IN HIDDEN DIV
# *********************************************************************************************************
@app.callback(Output('dataset-value', 'children'), 
            [Input('shape-dropdown', 'value'), Input('sample-slider', 'value'), Input('cluster-slider', 'value'), Input('regenData-button', 'n_clicks')])
def update_dataset(sampleShape, sampleSize, n_clusters, regenData_n_clicks):
    # CREATE DATASET
    X = create_dataset(shape=sampleShape, sampleSize=sampleSize, n_clusters=n_clusters)
    X = json.dumps(X.tolist())
    return X
# *********************************************************************************************************

# *********************************************************************************************************
#                        CALLBACK 2 : UPDATE CENTROIDS AND STORE IN HIDDEN DIV
# *********************************************************************************************************
@app.callback(Output('kmeansCentroids-value', 'children'), 
            [Input('init-dropdown', 'value'), Input('centroid-slider', 'value'), Input('dataset-value', 'children'), Input('regenCentroids-button', 'n_clicks')])
def update_kmeans_centroids(initMethod, n_centroids, dataset, regenCentroids_n_clicks):
    X = np.array(json.loads(dataset))
    centroids = init_centroids(X, k=n_centroids, initMethod=initMethod)
    centroids = json.dumps(centroids.tolist())
    return centroids
# *********************************************************************************************************

# *********************************************************************************************************
#                         CALLBACK 3 : UPDATE K-MEANS FRAMES, INERTIA, SILHOUETTE
# *********************************************************************************************************
globalKmeansFramesCounter = -1
globalFrames = []
globalInertia = []
globalSilhouette = tuple()
@app.callback(Output('kmeansFrames-counter', 'children'), 
            [Input('dataset-value', 'children'), Input('kmeansCentroids-value', 'children'), Input('iter-slider', 'value')])
def update_kmeansFrames(dataset, kmeans_centroids, max_iter):
    global globalKmeansFramesCounter, globalFrames, globalInertia, globalSilhouette

    # UPDATE COUNTER
    globalKmeansFramesCounter = globalKmeansFramesCounter + 1
    
    # LOAD DATASET & CENTROIDS **************************
    X = np.array(json.loads(dataset))
    centroids = np.array(json.loads(kmeans_centroids))
    n_centroids = centroids.shape[0]
    labels = [-1]*X.shape[0]
    # RUN K-MEANS ***************************************************
    inertia_hist = []
    kmeans_frames = []
    kmeans_frames.append(make_kmeans_viz_data(X, labels, centroids, clusColorMap))
    for i in range(max_iter):
        # Expectation Step
        labels =   Kmeans_EStep(X, centroids)
        kmeans_frames.append(make_kmeans_viz_data(X, labels, centroids, clusColorMap))
        # Maximization Step
        centroids = Kmeans_MStep(X, centroids, labels, n_centroids)
        kmeans_frames.append(make_kmeans_viz_data(X, labels, centroids, clusColorMap))
        # Compute Inertia
        inertia = 0
        for j in range(n_centroids):
            inertia = inertia + np.power(np.linalg.norm(X[labels==j,:]-centroids[j], axis=1), 2).sum()
        inertia_hist.append(inertia)
    globalInertia = inertia_hist
    globalInertia = (globalInertia, KMeans(n_clusters=len(centroids)).fit(X).inertia_)
    # COMPUTE SILHOUETTE ********************************************
    silhouetteValues = silhouette_samples(X, labels)
    globalSilhouette = ({k:silhouetteValues[labels==k] for k in range(n_centroids)}, silhouette_score(X, labels))
    # CREATE FRAMES *************************************************
    globalFrames = [{'data':kmeans_frames[0], 'layout':{**layout, 'title':'Intialization...'}}]
    globalFrames = globalFrames +  [{'data':d, 'layout':{**layout, 'title':'Step {} : {}'.format(idx//2+1, 'Expectation')}} if idx%2==0  
                                    else {'data':d, 'layout':{**layout, 'title':'Step {} : {}'.format(idx//2+1, 'Maximization')}}
                                    for idx,d in enumerate(kmeans_frames[1:])]

    return globalKmeansFramesCounter
# *********************************************************************************************************

# *********************************************************************************************************
#                                   CALLBACK 4 : UPDATE K-MEANS GRAPH
# *********************************************************************************************************
globalCurrentStep = 0
globalPrevClicks = 0
globalNextClicks = 0
globalRestartClicks=0
globalNumIntervals = 0
globalFramesCounter = 0

layout = dict(
    xaxis = dict(zeroline=False, showgrid=False, showline=True, showticklabels=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', range=[-17,17]),
    yaxis = dict(zeroline=False, showgrid=False, showline=True, showticklabels=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', range=[-17,17]),
    height = 450,
    margin = {'t':10, 'b':30, 'l': 40}, 
    plot_bgcolor = "rgba(75,77,85,1)"  
)

@app.callback(Output('kmeans-graph', 'figure'), 
            [
            Input('nextStep-button', 'n_clicks'), Input('prevStep-button', 'n_clicks'), Input('restart-button', 'n_clicks_timestamp'), 
            Input('kmeansFrames-counter', 'children'), Input('interval-component', 'n_intervals')
            ])
def update_kmeans_graph(nextStep_n_clicks, prevStep_n_clicks, restart_n_clicks, frames_counter, n_intervals):
    global globalPrevClicks, globalNextClicks, globalCurrentStep, globalRestartClicks, globalNumIntervals, globalFramesCounter

    if prevStep_n_clicks == None: prevStep_n_clicks=0
    if nextStep_n_clicks == None: nextStep_n_clicks=0
    if restart_n_clicks == None: restart_n_clicks=0
    if n_intervals == None: n_intervals = 0


    if (globalRestartClicks != restart_n_clicks) or (globalFramesCounter != frames_counter):
        globalRestartClicks = restart_n_clicks
        globalFramesCounter = frames_counter
        globalCurrentStep = 0
        d = globalFrames[globalCurrentStep]['data']
        fig = dict(data=d, layout=layout)
        return fig

    elif (globalNextClicks != nextStep_n_clicks) or (globalNumIntervals != n_intervals):
        globalNextClicks = nextStep_n_clicks
        globalNumIntervals = n_intervals
        globalCurrentStep = min(globalCurrentStep + 1, len(globalFrames)-1)
        d = globalFrames[globalCurrentStep]['data']
        fig = dict(data=d, layout=layout)
        return fig

    elif globalPrevClicks != prevStep_n_clicks:
        globalPrevClicks = prevStep_n_clicks
        globalCurrentStep = max(globalCurrentStep - 1, 0)
        d = globalFrames[globalCurrentStep]['data']
        fig = dict(data=d, layout=layout)
        return fig     
 
    d = globalFrames[globalCurrentStep]['data']
    fig = dict(data=d, layout=layout)
    return fig
# *********************************************************************************************************

# *********************************************************************************************************
#                           CALLBACK 5 : UPDATE STEP/ITERATION TEXT
# *********************************************************************************************************
@app.callback(Output('iter-text', 'children'), [Input('kmeans-graph', 'figure')])
def update_iter_text(kmeans_fig):
    text = globalFrames[globalCurrentStep]['layout']['title']
    return text 
# *********************************************************************************************************

# *********************************************************************************************************
#                           CALLBACK 6 : DISABLE NB CLUSTERS FOR CERTAIN SHAPES
# *********************************************************************************************************
@app.callback(Output('cluster-slider', 'disabled'), [Input('shape-dropdown', 'value')])
def disable_component(shape):
    if shape in ['moon', 'circle', 'noStructure']:
        return True
    return False
# *********************************************************************************************************

# *********************************************************************************************************
#                       CALLBACK 7 : CONTROL ANIMATION USING PLAY/PAUSE BUTTONS
# *********************************************************************************************************
globalPlayClicks = 0
globalPauseClicks = 0
@app.callback(Output('interval-component', 'interval'), [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')])
def play_kmeans(play_clicks, pause_clicks):
    global globalPlayClicks, globalPauseClicks, currentStepIter
    if play_clicks==None: play_clicks=0
    if pause_clicks==None: pause_clicks=0
    if globalPlayClicks != play_clicks:
        globalPlayClicks = play_clicks
        return 1000
    return 3600*1000
# *********************************************************************************************************

# *********************************************************************************************************
#                                  CALLBACK 8 : STATE OF PLAY BUTTON
# *********************************************************************************************************
@app.callback(Output('play-button', 'style'), [Input('interval-component', 'interval')])
def update_play_state(interval):
    if interval == 1000:
        style = {'background-color':'#33C3F0', 'border-color':'white', 'marginLeft':45, 'marginTop':30, 'marginBottom':10, 'marginRight':5, 'width':60, 'font-size':10, 'color':'white', 'text-align': 'center', 'padding': 0}
    else:
        style = {'marginLeft':45, 'marginTop':30, 'marginBottom':10, 'marginRight':5, 'width':60, 'font-size':10, 'text-align': 'center', 'padding': 0}
    return style
# *********************************************************************************************************

# *********************************************************************************************************
#                                       CALLBACK 9 : INERTIA GRAPH
# *********************************************************************************************************
layoutInertia = dict(
    #title = 'K-Means Cost Function',
    xaxis = dict(title='Iteration', zeroline=False, showgrid=False, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks'),
    yaxis = dict(title='Inertia', zeroline=False, showgrid=True, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', gridcolor="silver", tickformat='.0f'),
    height = 320,
    margin = {'t':10, 'l':60, 'r':10, 'b':40},
    showlegend = False,
    plot_bgcolor = "rgba(75,77,85,1)"
)

@app.callback(Output('inertia-graph', 'figure'), [Input('kmeansFrames-counter', 'children')])
def update_inertia_graph(frames_counter):
    data1 = go.Scatter(
        x = list(range(1, len(globalInertia[0])+1)), 
        y = globalInertia[0],
        mode = 'markers+lines',
        marker = dict(color='white', size=10, line = dict(width=2, color='rgb(205,92,92,1)')),
        line = dict(color='rgba(205,92,92,1)'),
        name = 'Current Initialization')
    
    data2 = go.Scatter(
        x = [len(globalFrames)//2-1],
        y = [globalInertia[1]+0.1*(globalInertia[0][0]-globalInertia[1])],
        mode = 'text',
        text = 'Global Minimum',
        textfont = dict(
            color = "white"
        )
    )

    layoutInertia['shapes'] = [dict(type='line', x0=0, y0=globalInertia[1], x1=len(globalFrames)//2+1, y1=globalInertia[1], line={'color':'white', 'width':1, 'dash':'dot'})]
    fig = dict(data=[data1, data2], layout=layoutInertia)
    return fig
# *********************************************************************************************************

# *********************************************************************************************************
#                                   CALLBACK 10 : UPDATE SILHOUETTE GRAPH
# *********************************************************************************************************
layoutSilhouette = dict(
    xaxis = dict(title='Silhouette Coefficient', zeroline=False, showgrid=True, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', gridcolor="silver"),
    yaxis = dict(title='Cluster', showticklabels=False, zeroline=False, showgrid=False, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks'),
    height = 320,
    margin = {'t':20, 'l':50, 'r':10, 'b':40},
    showlegend = False,
    plot_bgcolor = "rgba(75,77,85,1)"
)

@app.callback(Output('silhouette-graph', 'figure'), [Input('kmeansFrames-counter', 'children')])
def update_silhouette_graph(frames_counter):
    data = []
    y_lower = 5
    nCentroids = len(globalSilhouette[0])
    sampleSize = len([x for sublist in globalSilhouette[0].values() for x in sublist]) 
    for i in range(nCentroids):
        silhouetteValues = globalSilhouette[0][i]
        silhouetteValues.sort()
        y_upper = y_lower + len(silhouetteValues)
        filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                 x=silhouetteValues,
                                 mode='lines',
                                 showlegend=False,
                                 line=dict(width=1, color=clusColorMap[i]),
                                 fill='tozerox',
                                 fillcolor= lineColors[i])
        data.append(filled_area)
        y_lower = y_upper + 5

    trace = go.Scatter(
        x = [globalSilhouette[1]-.1],
        y = [.2*sampleSize],
        mode = 'text',
        text = 'Avg. Score',
        textfont = dict(
            color="white"
        )
    )
    data.append(trace)
    layoutSilhouette['yaxis']['range'] = [0, sampleSize+5*(nCentroids+1)]
    layoutSilhouette['shapes'] = [dict(type='line', x0=globalSilhouette[1], y0=0, x1=globalSilhouette[1], y1=sampleSize+6*nCentroids, line={'color':'white', 'width':2, 'dash':'dot'})]

    fig = dict(data=data, layout=layoutSilhouette)
    return fig
# *********************************************************************************************************


# ***************************************
#               EXECUTION
# ***************************************
if __name__ == '__main__':
    app.run_server(debug=True)