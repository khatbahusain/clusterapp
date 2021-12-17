import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px

import vectorizer
import clusterer



app = dash.Dash(__name__, external_stylesheets=['style.css'])


app.layout = html.Div([
    
    html.H2("ClusterAPP 📊", style={"color": "#FFA500", "line-height": "0.1px"}),
    #html.Img(src='https://mathbabe.files.wordpress.com/2012/09/cluster.png', style={'width': '50px', 'height': '50px'}),
    html.P("Cluster similar sentences", style={"margin-left": "50px"}),
    html.Hr(),


    html.Div(html.Div([dcc.Textarea(id='textarea-state-example',
                                    placeholder='Ajoutez des phrases...',
                                    value='',
                                    style={'width': '33.33%', 'height': 250, "display": "inline-block", "margin": "15px"}),

            
            html.Div([html.Span('Embedding Algorithm:'),
                      dcc.RadioItems(options=[{'label': 'Term frequency–Inverse document frequency', 'value': 'tfidf'},{'label': 'Universal Sentence Encoder Multilingual Large', 'value': 'use'}],value='tfidf', id="algovec",), 
                      html.Br(),
                      html.Span('Clustering Algorithm:'), 
                      dcc.RadioItems(options=[{'label': 'Kmeans', 'value': 'kmeans'}],value='kmeans'), 
                      html.Br(),html.Br(),
                
                      html.P("Choose a cluster number or a range to find the best number automatically"),
                      
                      html.Div(dcc.RangeSlider(
                               id='range_values', 
                               min=2,
                               max=30,
                               value=[5, 8],
                               tooltip={"placement": "bottom", "always_visible": True}), style={})],style={"display": "inline-block", 'vertical-align': 'top', 'height': 200, "margin": "15px"})])),


    html.Button('RUN', id='textarea-state-example-button', n_clicks=0, style={"margin": "15px"}),
    html.Br(),html.Br(),

    html.Div(id='textarea-state-example-output'), 

    ], 
    
    
    style={"margin": "auto", "background": "white", "padding": "10px", "max-width" : "70%"})


@app.callback(
    Output('textarea-state-example-output', 'children'),
    Input('textarea-state-example-button', 'n_clicks'),
    State('textarea-state-example', 'value'),
    State('algovec', 'value'),
    State('range_values', 'value'))



def main(n_clicks, corpus, algovec, range_values):

    if n_clicks > 0:
       
        corpus = corpus.split('\n')

        if len(corpus)<=1:
            return "Ajoutez d'abord quelques phrases pour les regrouper"

       
        ## Vectorizer
        if algovec == 'tfidf':
            vectors = vectorizer.tfidf(corpus)
        elif algovec == 'use':
            vectors = vectorizer.useml(corpus)


        # if clusterer
        if range_values[1] >= len(corpus):

            return f"Max value of clusters should be less than the number of sentences : {len(corpus)}" 


        if range_values[0] == range_values[1]:

            
            n_cluster = range_values[0]
            clusters = clusterer.kmeans(vectors, n_cluster)

            d = {'Sentence': corpus, 'Cluster': clusters}
            df = pd.DataFrame(data=d)

            result = dash_table.DataTable(
                        filter_action='native',
                        export_format='csv',
                        export_headers='display',
                        merge_duplicate_headers=True,
                        editable=True,
                        row_deletable=True,
                        columns=[{"name": i, "id": i, 'deletable': True, 'renamable': True} for i in df.columns],
                        data=df.to_dict('records'),
                        page_size=15)

            return result


        elif range_values[0] != range_values[1]:
            
            clusters, km_silhouette, n_cluster = clusterer.silhouette(vectors, range_values)

       
            _ = pd.DataFrame({'k':[i for i in range(range_values[0], range_values[1])], 'silhouette':km_silhouette})

            fig = px.line(_, x='k', y='silhouette', title="Silhouette Score", markers=True)

            fig.add_annotation(x=n_cluster, y=km_silhouette[[i for i in range(range_values[0], range_values[1])].index(n_cluster)], text="Best n of Cluster", 
            showarrow=True, arrowhead=1)


            silhouette = dcc.Graph(figure=fig)

            d = {'Sentence': corpus, 'Cluster': clusters}
            df = pd.DataFrame(data=d)
            df['Cluster'] = df['Cluster'].astype(int)
            df = df.sort_values('Cluster')

            result = dash_table.DataTable(
                        filter_action='native',
                        export_format='csv',
                        export_headers='display',
                        merge_duplicate_headers=True,
                        editable=True,
                        row_deletable=True,
                        columns=[{"name": i, "id": i, 'deletable': True, 'renamable': True} for i in df.columns],
                        data=df.to_dict('records'),
                        page_size=15,
                        style_table={})

            return silhouette, result


if __name__ == '__main__':
    app.run_server(debug=False)

## select coulumn 