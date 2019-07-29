# *********************************************************************************************************************
#                                   K-MEANS CLUSTERING : INTERACTIVE VISUALIZATION
# *********************************************************************************************************************


# ***************************************
#                IMPORTS
# ***************************************
import numpy as np
from random import shuffle
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import os
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from HelpFunctions import dist, create_dataset, init_centroids, Kmeans_EStep, Kmeans_MStep, make_kmeans_viz_data

# ***************************************
#              INITIALIZATION
# ***************************************
app = dash.Dash(__name__)

# ***************************************
#            CSS CUSTOMIZATION
# ***************************************
#app.css.append_css({
 #   "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
#})

# ***************************************
#              APPLICATION
# ***************************************

## COLOR PALETTES
clusColorMap = {0:'#6395EC', 1:'#FFA27A', 2:'#DDA618', 3:'#DFA5DD', 4:'#61CCAB', 5:'#CD5C5C', 6:'#719D2F', 7:'#8E4D2C', 8:'#472F63'}
lineColors = {0: 'rgba(99,149,236.4)', 1: 'rgba(255,162,122,.4)', 2: 'rgba(221,166,24.4)', 3: 'rgba(223,165,221,.4)', 
              4: 'rgba(97,204,171,.4)', 5: 'rgba(205,92,92,.4)', 6: 'rgba(113,157,47,.4)', 7:'rgba(142,77,44,.4)', 
              8: 'rgba(71,47,99,.4)'}

# LAYOUT
layout = dict(
    #title = 'K-Means Scatter Plot',
    xaxis = dict(zeroline=False, showgrid=False, showline=True, showticklabels=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', range=[-17,17]),
    yaxis = dict(zeroline=False, showgrid=False, showline=True, showticklabels=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', range=[-17,17]),
    height = 450,
    #width = 600,
    margin = {'t':10, 'b':30},
    #plot_bgcolor = "black"  
    plot_bgcolor = "rgba(75,77,85,1)"  
)


#figure = dict(data=frames[0]['data'], layout=frames[0]['layout'])

## BUILD DASH APPLICATION
    ## APP LAYOUT
app.layout = html.Div(children=[

    # COLUMN 1 (TITLE + PARAMETERS BLOCK + K-MEANS GRAPH)
    html.Div([
        # ROW 1 (TITLE)
        html.Div([
            html.H1('K-means : Interactive Visualization', style = {'color' : '#4B4D55', 'font-family':'savoye LET', 'fontSize' : 80, 'fontWeight' : 100, 'textAlign' : 'center', 'marginTop':'2%', 'marginBottom':'0%'})
        ], style = {'height' : '12.5%', 'border-radius':10, 'borderStyle' : 'solid', 'borderWidth' : 4, 'borderColor' : 'black', 'marginRight':'2%', 'marginBottom':'2%', 'background-color':'white'}
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
                    ], style = {'marginLeft': 10, 'marginRight': 10})

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
                ]),
            ], className='three columns', style = {'background-color':'white', 'height' : '670', 'borderStyle' : 'solid', 'border-radius':10, 'borderWidth' : 4, 'borderColor' : 'black', 'marginRight':'0%'}),
            # ROW 2 SECOND PART (GRAPH)
            html.Div([
                html.H6('K-MEANS SCATTER PLOT', style = {'text-align':'center', 'marginTop' : 30, 'marginBottom':5, 'fontSize':15, 'fontWeight':800, 'color' : '#4B4D55'}),

                html.Button('Play', id='play-button', style={'marginLeft':80, 'marginTop':30, 'marginBottom':10, 'marginRight':5, 'width':60, 'font-size':10, 'text-align': 'center', 'padding': 0}),
                html.Button('Pause', id='pause-button', style={'marginRight':5, 'width':70, 'font-size':10, 'text-align':'center', 'padding': 0}),
                html.Button('<<', id='prevStep-button', style={'marginRight':5, 'width':40, 'font-size':10, 'text-align':'center', 'padding': 0}),
                html.Button('>>', id='nextStep-button', style={'marginRight':5, 'width':40, 'font-size':10, 'text-align':'center', 'padding': 0}),
                html.Button('Restart', id='restart-button', style={'width':90, 'font-size':10, 'text-align':'center', 'padding': 0}),

                html.Button(id='iter-text', disabled=True, style={'marginLeft':'10%', 'width':180, 'background-color':'#4B4D55', 'pointer-events': 'none', 'color':'white', 'font-size':10, 'text-align':'center', 'padding': 0}),
                #html.Div(id='iter-text', style = {'display':'inline-block', 'width':250, 'height':30, 'marginLeft':20, 'border-style':'solid', 'border-color':'#888888', 'border-width':1}),

                dcc.Graph(id='kmeans-graph', animate=True, config={'displayModeBar': False}),
                dcc.Interval(id='interval-component', interval=3600*1000, n_intervals=0),
                html.Img(src=app.get_asset_url('Signature_Logo.png'), style={'width':70, 'bottom':25, 'left':'92%', 'position':'relative'})#775

            ], className = 'nine columns', style = {'background-color':'white', 'height' : '670', 'border-width':4, 'borderStyle' : 'solid', 'border-radius':10, 'borderColor' : 'black', 'marginLeft':'2%'}),
 
        ])
        
    ], className = 'eight columns', style = { 'marginLeft':30, 'marginRight':0}),
    # COLUMN 2 (COST FUNCTION GRAPH + SILHOUETTE GRAPH)
    html.Div([
        html.Div([
            html.H6('K-MEANS COST FUNCTION', style = {'text-align':'center', 'marginTop' : 15, 'marginBottom':5, 'fontSize':15, 'fontWeight':800, 'color' : '#4B4D55'}),
            dcc.Graph(id='inertia-graph', config={'displayModeBar': False}),
        ], style = {'background-color':'white', 'height' : 390, 'marginBottom':20, 'border-radius':10, 'borderStyle' : 'solid', 'borderWidth' : 4, 'borderColor' : 'black'}),
        
        html.Div([
            html.H6('SILHOUETTE ANALYSIS', style = {'text-align':'center', 'marginTop' : 15, 'marginBottom':5, 'fontSize':15, 'fontWeight':800, 'color' : '#4B4D55'}),
            dcc.Graph(id='silhouette-graph', config={'displayModeBar': False})
        ], style = {'background-color':'white', 'height' : 390, 'border-radius':10, 'borderStyle' : 'solid', 'borderWidth' : 4, 'borderColor' : 'black'}),
    ], className = 'four columns', style={ 'height':800, 'marginLeft':0, 'marginRight':0}),

    # COPYRIGHT
    html.Div([
        html.H6('Copyright © 2019  Samy TAFASCA', style={'color':'white', 'font-size':10, 'text-align':'right'})
    ], className= 'twelve columns')
]) 



# ***************************************
#               CALLBACKS
# ***************************************

# CALLBACK 1 : UPDATE K-MEANS GRAPH
prevClicks = 0
nextClicks = 0
restartClicks=0
currentStep = 0
globalSampleShape = 'gmm'
globalSampleSize=200
globalNClusters = 3
globalNCentroids = 3
globalMaxIter = 10
globalInitMethod = 'random'
globalX = np.empty((globalSampleSize, 2))
globalFrames = []
globalInertia = []
globalSilhouette = tuple()

@app.callback(Output('kmeans-graph', 'figure'), 
            [Input('nextStep-button', 'n_clicks'), Input('prevStep-button', 'n_clicks'), Input('restart-button', 'n_clicks'), 
            Input('shape-dropdown', 'value'), Input('sample-slider', 'value'), Input('cluster-slider', 'value'), 
            Input('init-dropdown', 'value'), Input('centroid-slider', 'value'), Input('iter-slider', 'value'),
            Input('interval-component', 'n_intervals')
            ])
def update_kmeans_graph(nextStep_n_clicks, prevStep_n_clicks, restart_n_clicks, 
                        sampleShape, sampleSize, n_clusters,
                        initMethod, n_centroids, max_iter,
                        n_intervals):
    global prevClicks, nextClicks, currentStep, restartClicks, globalSampleShape, globalSampleSize, globalNClusters, globalNCentroids, globalInitMethod, globalMaxIter, globalX, globalFrames, globalInertia, globalSilhouette

    if prevStep_n_clicks == None: prevStep_n_clicks=0
    if nextStep_n_clicks == None: nextStep_n_clicks=0
    if restart_n_clicks == None: restart_n_clicks=0

    if (globalSampleSize != sampleSize) or (n_clusters != globalNClusters) or (sampleShape != globalSampleShape) or globalFrames == []:
        # UPDATE GLOBAL VARIABLES
        globalSampleSize = sampleSize
        globalNClusters = n_clusters
        globalSampleShape = sampleShape 
        currentStep = 0
        # CREATE DATASET
        X = create_dataset(shape=sampleShape, sampleSize=sampleSize, n_clusters=n_clusters)
        globalX = X.copy()
        # INITIALIZE K-MEANS
        centroids = init_centroids(X, k=n_centroids, initMethod=initMethod)
        labels = [-1]*X.shape[0]
        # RUN K-MEANS
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
        globalInertia = (globalInertia, KMeans(n_clusters=len(centroids)).fit(globalX).inertia_)
        # COMPUTE SILHOUETTE
        silhouetteValues = silhouette_samples(X, labels)
        globalSilhouette = ({k:silhouetteValues[labels==k] for k in range(n_centroids)}, silhouette_score(X, labels))

        # CREATE FRAMES
        globalFrames = [{'data':kmeans_frames[0], 'layout':{**layout, 'title':'Intialization...'}}]
        globalFrames = globalFrames +  [{'data':d, 'layout':{**layout, 'title':'Step {} : {}'.format(idx//2+1, 'Expectation')}} if idx%2==0  
                                        else {'data':d, 'layout':{**layout, 'title':'Step {} : {}'.format(idx//2+1, 'Maximization')}}
                                        for idx,d in enumerate(kmeans_frames[1:])]
        # RETURN FIGURE
        d = globalFrames[0]['data']
        fig = dict(data=d, layout=layout)
        return fig

    elif (n_centroids != globalNCentroids) or (initMethod != globalInitMethod) or (max_iter != globalMaxIter):
        # UPDATE GLOBAL VARIABLES
        currentStep = 0
        globalNCentroids = n_centroids
        globalMaxIter = max_iter
        globalInitMethod = initMethod
        # INITIALIZE K-MEANS
        centroids = init_centroids(globalX, k=n_centroids, initMethod=initMethod)
        labels = [-1]*globalX.shape[0]
        # RUN K-MEANS
        inertia_hist = []
        kmeans_frames = []
        kmeans_frames.append(make_kmeans_viz_data(globalX, labels, centroids, clusColorMap))
        for i in range(max_iter):
            # Expectation Step
            labels =   Kmeans_EStep(globalX, centroids)
            kmeans_frames.append(make_kmeans_viz_data(globalX, labels, centroids, clusColorMap))
            # Maximization Step
            centroids = Kmeans_MStep(globalX, centroids, labels, n_centroids)
            kmeans_frames.append(make_kmeans_viz_data(globalX, labels, centroids, clusColorMap))
            # Compute Inertia
            inertia = 0
            for j in range(n_centroids):
                inertia = inertia + np.power(np.linalg.norm(globalX[labels==j,:]-centroids[j], axis=1), 2).sum()
            inertia_hist.append(inertia)
        globalInertia = inertia_hist
        globalInertia = (globalInertia, KMeans(n_clusters=len(centroids)).fit(globalX).inertia_)
        # COMPUTE SILHOUETTE
        silhouetteValues = silhouette_samples(globalX, labels)
        globalSilhouette = ({k:silhouetteValues[labels==k] for k in range(n_centroids)}, silhouette_score(globalX, labels))
        # CREATE FRAMES
        globalFrames = [{'data':kmeans_frames[0], 'layout':{**layout, 'title':'Intialization...'}}]
        globalFrames = globalFrames +  [{'data':d, 'layout':{**layout, 'title':'Step {} : {}'.format(idx//2+1, 'Expectation')}} if idx%2==0  
                                        else {'data':d, 'layout':{**layout, 'title':'Step {} : {}'.format(idx//2+1, 'Maximization')}}
                                        for idx,d in enumerate(kmeans_frames[1:])]
        # RETURN FIGURE
        d = globalFrames[0]['data']
        fig = dict(data=d, layout=layout)
        return fig

    if restartClicks < restart_n_clicks:
        restartClicks = restart_n_clicks
        currentStep = 0
        d = globalFrames[0]['data']
        fig = dict(data=d, layout=layout)
        return fig

    if (prevClicks != prevStep_n_clicks):
        prevClicks = prevStep_n_clicks
        d = globalFrames[max(0, currentStep-1)]['data']
        currentStep = max(0, currentStep-1)
        fig = dict(data=d, layout=layout)
        return fig   

    if (nextClicks != nextStep_n_clicks):
        nextClicks = nextStep_n_clicks
        d = globalFrames[min(2*max_iter, currentStep+1)]['data']
        currentStep = min(2*max_iter, currentStep+1)
        fig = dict(data=d, layout=layout)
        return fig     
 
    d = globalFrames[currentStep+1]['data']
    currentStep = currentStep+1
    fig = dict(data=d, layout=layout)
    return fig

# CALLBACK 2 : UPDATE ITERATION NUMBER TEXT
globalSampleShapeIter = 'gmm'
globalSampleSizeIter = 200
globalNClustersIter = 3
globalNCentroidsIter = 3
globalMaxIterationIter = 10
globalInitMethodIter = 'random'
restartClicksIter = 0
nextClicksIter = 0
prevClicksIter = 0
currentStepIter = 0

@app.callback(Output('iter-text', 'children'), 
            [Input('nextStep-button', 'n_clicks'), Input('prevStep-button', 'n_clicks'), Input('restart-button', 'n_clicks'), 
            Input('shape-dropdown', 'value'), Input('sample-slider', 'value'), Input('cluster-slider', 'value'), 
            Input('init-dropdown', 'value'), Input('centroid-slider', 'value'), Input('iter-slider', 'value'),
            Input('interval-component', 'n_intervals')
            ])
def update_iter_text(nextStep_n_clicks, prevStep_n_clicks, restart_n_clicks, 
                    sampleShape, sampleSize, n_clusters,
                    initMethod, n_centroids, max_iter,
                    n_intervals):
    global nextClicksIter, prevClicksIter, currentStepIter, restartClicksIter, globalMaxIterationIter, globalNCentroidsIter, globalInitMethodIter, globalNClustersIter, globalSampleSizeIter, globalSampleShapeIter
    
    if globalFrames == []:
        currentStepIter = 1
        return 'Initialization...'
    if nextStep_n_clicks == None: nextStep_n_clicks=0
    if prevStep_n_clicks == None: prevStep_n_clicks=0
    if restart_n_clicks == None: restart_n_clicks=0

    if(restartClicksIter != restart_n_clicks) or (globalMaxIterationIter != max_iter) or (globalInitMethodIter != initMethod) or (globalSampleShapeIter != sampleShape) or (globalNCentroidsIter != n_centroids) or (globalNClustersIter != n_clusters) or (globalSampleSizeIter != sampleSize):
        restartClicksIter = restart_n_clicks
        globalMaxIterationIter = max_iter
        globalInitMethodIter = initMethod
        globalNCentroidsIter = n_centroids
        globalNClustersIter = n_clusters
        globalSampleSizeIter = sampleSize
        globalSampleShapeIter = sampleShape
        currentStepIter = 1
        text = globalFrames[0]['layout']['title']
        return text 

    
    if(nextClicksIter != nextStep_n_clicks):
        nextClicksIter = nextStep_n_clicks
        text = globalFrames[currentStepIter]['layout']['title']
        currentStepIter = min(2*max_iter, currentStepIter+1)
        return text

    if(prevClicksIter != prevStep_n_clicks):
        prevClicksIter = prevStep_n_clicks
        text = globalFrames[max(0, currentStepIter-2)]['layout']['title']
        currentStepIter = max(1, currentStepIter-1)
        return text



    text = globalFrames[currentStepIter]['layout']['title']
    currentStepIter = currentStepIter+1
    return text 

# CALLBACK 3 : DISABLE NB CLUSTERS FOR CERTAIN SHAPES
@app.callback(Output('cluster-slider', 'disabled'), [Input('shape-dropdown', 'value')])
def disable_component(shape):
    if shape in ['moon', 'circle', 'noStructure']:
        return True
    return False

# CALLBACK 4 : PLAY/PAUSE BUTTONS
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


# CALLBACK 5 : INERTIA GRAPH
layoutInertia = dict(
    #title = 'K-Means Cost Function',
    xaxis = dict(title='Iteration', zeroline=False, showgrid=True, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks'),
    yaxis = dict(title='Inertia', zeroline=False, showgrid=True, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks'),
    height = 320,
    margin = {'t':10, 'l':60, 'r':20, 'b':40},
    showlegend = False
    #plot_bgcolor = "rgba(71,47,99,.6)" 
)
@app.callback(Output('inertia-graph', 'figure'), [Input('kmeans-graph', 'figure')])
def update_inertia_graph(kmeans_fig):
    data1 = go.Scatter(
        x = list(range(1, len(globalInertia[0])+1)), 
        y = globalInertia[0],
        mode = 'markers+lines',
        marker = dict(color='white', size=10, line = dict(width=2, color='rgb(205,92,92,.8)')),
        line = dict(color='rgba(205,92,92,.8)'),
        name = 'Current Initialization')
    
    data2 = go.Scatter(
        x = [globalMaxIter-1],
        y = [globalInertia[1]+0.1*(globalInertia[0][0]-globalInertia[1])],
        mode = 'text',
        text = 'Global Minimum'
    )

    layoutInertia['shapes'] = [dict(type='line', x0=0, y0=globalInertia[1], x1=globalMaxIter+1, y1=globalInertia[1], line={'color':'#4B4D55', 'width':2, 'dash':'dot'})]
    fig = dict(data=[data1, data2], layout=layoutInertia)
    return fig


# CALLBACK 6 : UPDATE SILHOUETTE GRAPH
layoutSilhouette = dict(
    xaxis = dict(title='Silhouette Coefficient', zeroline=False, showgrid=True, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks'),
    yaxis = dict(title='Cluster', showticklabels=False, zeroline=False, showgrid=True, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks'),
    height = 320,
    margin = {'t':10, 'l':60, 'r':20, 'b':40},
    showlegend = False
    #plot_bgcolor = "rgba(71,47,99,.6)" 
)
@app.callback(Output('silhouette-graph', 'figure'), [Input('kmeans-graph', 'figure')])
def update_silhouette_graph(kmeans_fig):
    data = []
    y_lower = 5
    for i in range(globalNCentroids):
        silhouetteValues = globalSilhouette[0][i]
        silhouetteValues.sort()
        y_upper = y_lower + len(silhouetteValues)
        filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                 x=silhouetteValues,
                                 mode='lines',
                                 showlegend=False,
                                 line=dict(width=0.5, color=clusColorMap[i]),
                                 fill='tozerox')
        data.append(filled_area)
        y_lower = y_upper + 5

    trace = go.Scatter(
        x = [globalSilhouette[1]-.1],
        y = [.2*globalSampleSize],
        mode = 'text',
        text = 'Avg. Score'
    )
    data.append(trace)
    layoutSilhouette['yaxis']['range'] = [0, globalSampleSize+5*(globalNCentroids+1)]
    layoutSilhouette['shapes'] = [dict(type='line', x0=globalSilhouette[1], y0=0, x1=globalSilhouette[1], y1=globalSampleSize+5*globalNCentroids, line={'color':'#4B4D55', 'width':2, 'dash':'dot'})]

    fig = dict(data=data, layout=layoutSilhouette)
    return fig



# ***************************************
#               EXECUTION
# ***************************************
if __name__ == '__main__':
    app.run_server(debug=True)