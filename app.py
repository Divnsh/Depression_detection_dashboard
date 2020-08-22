import dash
import dash_core_components as dcc
import dash_html_components as html
#import plotly.express as px
#import json
from cleaning import Cleaning
import os
import pandas as pd
#import dash_dangerously_set_inner_html
#import pyldavis_dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from roBERTamodel import Model
from more_exploration import format_topics_inputs
import pickle
import base64
import io
import flask
import glob

PLOTS_DIR='./Plots'
DATA_DIR='./Data'
assert os.path.exists(PLOTS_DIR) and os.path.exists(DATA_DIR), "Plots or data not present. First run the prerequisites!"

#external_stylesheets=['https://codepen.io/amyoshino/pen/jzXypZ.css']
external_stylesheets=['/assets/amyoshinopen.css']
app = dash.Dash(__name__, title = 'Depression detection', external_stylesheets=external_stylesheets)
server = app.server

colors = {
    'background': '#111111',
    'text': '#0392cf',
    'subtext': '#5166A9',
    'jellybean' : "#DE6356",
    'Mellow Apricot' : '#f7b172',
    'Moss Green' : '#83A14D',
    'Dark Cornflower Blue' : '#203d85'
}
# from bs4 import BeautifulSoup
# from ast import literal_eval
# soup = BeautifulSoup(open(os.path.join(PLOTS_DIR,'lda.html'), 'r').read(), 'html.parser')
# for line in str(soup.find_all('script')[0]).split('\n'):
#     if line[:3]=='var':
#         #myvars=line[50:len(line)-1]
#         #myvars = line[4:len(line)-1]
#         myvars=line

app.layout = html.Div(style={'background-image': 'url("/assets/stigma-depression-woman-illustration_1320WJR-1-1024x640.png")',
                        'bottom': "0", 'right': "0", 'left': "0", 'top': "0"}, children=[

        html.Div([
            html.H1(children='Depression detection through text',
                    style={
                                'textAlign': 'center',
                                'color': colors['text'],
                            }
                    , className = "twelve columns"),

            html.Div(children='See if your text indicates signs of depression.',
                        style={
                                'textAlign': 'left',
                                'color': colors['Mellow Apricot'],
                                'fontSize': 18,
                                'margin-left':5
                            }
                    , className = 'eight columns'),

            html.Button('Reset', id='reset_button', n_clicks=0, style={'height':"50px", 'width':"200px",
                                                                       'float':"right",
                                                                       'color':"white"}, className = 'five columns')

            ],className = 'row'),
        html.Br(),
        html.Br(),
        html.Div([
            html.Div([
                html.P("Major depression topics from subreddits:",
                        style={ 'textAlign': 'center',
                                'color': colors['jellybean'],
                                 'font-weight': 'bold',
                            },className='ten columns'),
                 #pyldavis_dash.pyLDAvis(id='lda_graph', data=myvars)
                #dash_dangerously_set_inner_html
                html.Iframe(src=app.get_asset_url('lda.html'),
                style={'width': "100%",'height':"900px",'background-color': 'rgba(230, 230, 230, 0.9)'})
            ],className='ten columns'),

            html.Div([
                html.Label("Dominant depression topic(s) in your text",style={
                                'textAlign': 'center',
                                'color': colors['jellybean'],
                                'font-weight': 'bold'
                            }),
                dcc.Textarea(id="topics_out", style={'whiteSpace':'pre-line','height':"600px",'width':"100%",
                                                     'text-align': "justify"}),
                html.Img(id='result_image', style={'width':"100%",'height':"300px"})

            ],className='two columns')
        ],className='row'),

        html.Div([
            html.Div([
                html.Label("Let's try it out: ", style={'textAlign':"center",'color':colors['Mellow Apricot'],
                                            'font-weight': 'bold','background-color': 'rgba(255, 255, 255, 0.95)'}),
                dcc.Textarea(id = "input_text", placeholder="Enter your text here.", style={'width':"100%",
                                                                                            'height':"230px"}),

                html.Div([
                    html.Button('Go', id='input_text_button', n_clicks=0, style={'color':colors['text'],
                                                                                 'height':"50px",
                                                                                 'width':"400px",
                                                                                 'float':"right",
                                                                                 'position':'sticky'},
                                ),
                    dcc.Upload(
                            id='upload_data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '50px',
                                'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '2px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'color' : 'white',
                                'float':"left",
                                'width':"400px"
                            }, multiple=True),
                    html.Ul(id="output_results",style={'color':"white"}),
                ], className='row')
            ], className='eight columns'),
            html.Div([
                #html.P("Probability of depression: ", style={'color':colors['Moss Green'],'textAlign':"center"}),
                dcc.Graph(id="Indicator",style={'width':"100%","height":"300px"})
            ], className='four columns')
        ], className='row')

], className='twelve columns')

roberta=Model()

@app.callback(
    Output('result_image', 'src'),
    [Input('input_text_button', 'n_clicks')],
    [State('input_text', 'value')]
)
def result_img(n_clicks,value):
    if os.path.exists('./score.pkl'):
        os.remove('./score.pkl')
    if n_clicks > 0:
        if value is not None:
            if len(value) > 4:
                cleaned = Cleaning(pd.DataFrame({'Text': [str(value)]}),0)
                cleaned.to_csv(os.path.join(DATA_DIR,'input_cleaned.csv'),header=True,index=False)
                pred = roberta.get_prediction(cleaned)[0]
                if pred=="GPU unavailable":
                    score=200
                    with open('score.pkl', 'wb') as f:
                        pickle.dump(score, f)
                    return '/assets/thinking.jpg'
                score = int(round(pred * 100, 0))
                with open('score.pkl', 'wb') as f:
                    pickle.dump(score, f)
                if score>50:
                    return "/assets/sad.jpg"
                else:
                    return "/assets/happy.jpg"
            else:
                return '/assets/thinking.jpg'
        else:
            return '/assets/thinking.jpg'
    else:
        return '/assets/logo.png'

@app.callback(
    Output('topics_out', 'value'),
    [Input('result_image', 'src')]
)
def get_topics(src):
    if src!='/assets/logo.png':
        try:
            with open('score.pkl', 'rb') as f:
                score=pickle.load(f)
            cleaned=pd.read_csv(os.path.join(DATA_DIR,'input_cleaned.csv'))
            if score>50 and score<101:
                sent_topics_df=format_topics_inputs(cleaned)
                topics=[str(int(e)) for e in list(sent_topics_df['Dominant_Topic'])]
                contrib=[str(c) for c in list(sent_topics_df['Perc_Contribution'])]
                outtext="Top 3 dominant topics and their weightages (0-1) in your text are:\n\n\
                Topic No. "+topics[0]+"\nweightage: "+contrib[0]+"\n\n\
                Topic No. " + topics[1] + "\nweightage: " + contrib[1] + "\n\n\
                Topic No. " + topics[2] + "\nweightage: " + contrib[2] + "\n\n\
                \nYou can explore these topics further in the figure to the left."
            elif score<=50:
                outtext = "Your text does not indicate depression \U0001F643."
            else:
                outtext = "😕 Sorry for the inconvenience. GPU unavailable. Try another time."
                return outtext
        except:
            outtext="\U0001F610 You may try entering a longer text."
        return outtext
    else:
        return "You will see relevant topics here once you enter your text below.\n\
                \nYou can also provide a csv file with text data.\n\
                \nFetching results may take 10-20 seconds. Please be patient."


@app.callback(
    Output('topics_out','style'),
    [Input('topics_out','value')]
)
def color_text(value):
    out=value
    if 'Topic No.' in out:
        color = colors['Dark Cornflower Blue']
    else:
        color = colors['Moss Green']
    return {'whiteSpace':'pre-line','height':"600px",'width':"100%",'color':color}


@app.callback(
    Output('Indicator','figure'),
    [Input('result_image', 'src')]
)
def set_prediction(src):
    try:
        with open('score.pkl', 'rb') as f:
            score = pickle.load(f)
        if score==200:
            score=50
    except:
        score=50
    fig_prob = go.Figure(go.Indicator(
    domain={'x': [0, 1], 'y': [0, 1]},
    value=score,
    mode="gauge+number+delta",
    title={
        'text': "<span style='color:" + colors['text'] + "' > <b>Score (>50 indicates depression)</b> </span>"},
    delta={'reference': 50},
    gauge={'axis': {'range': [None, 100]},
           'steps': [
               {'range': [0, 50], 'color': "lightgray"},
               {'range': [50, 100], 'color': "gray"}],
           }))
    fig_prob.update_layout(paper_bgcolor='rgba(230, 230, 230, 0.95)')
    return fig_prob

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')), names=['Text'])
        print(df.head())
    elif 'xls' in filename or 'xlsx' in filename:
        df = pd.read_excel(io.BytesIO(decoded), names=['Text'])
    df['idx']=list(range(0,len(df)))
    df1=df.copy()
    df2=Cleaning(df,0)
    df2 = roberta.get_prediction_bulk(df2)
    df2 = df2.merge(df1,on='idx',how='right')
    return df2

@app.server.route('/dash/urlToDownload')
def download_csv():
    value = flask.request.args.get('value')
    # create a dynamic csv or file here using `StringIO`
    # (instead of writing to the file system)
    df=pd.read_csv('./download/' + value)
    str_io = io.StringIO()
    df.to_csv(str_io,index=False)
    mem = io.BytesIO()
    mem.write(str_io.getvalue().encode('utf-8'))
    mem.seek(0)
    str_io.close()
    return flask.send_file(mem,
                           mimetype='text/csv',
                           attachment_filename=value,
                           as_attachment=True)


@app.callback(
    Output('output_results', 'children'),
    [Input('upload_data','filename')],
    [State('upload_data', 'contents')]
)
def fetch_results_csv(filename,contents):
    if contents is not None and filename is not None:
        fileList = glob.glob('./download/result*.csv')
        for f in fileList:
            try:
                os.remove(f)
            except:
                pass
        i=0
        refslist=[]
        for fname,data in zip(filename,contents):
            try:
                df=parse_contents(data, fname)
            except Exception as e:
                print(e)
                return 'There was an error processing this file. Please provide a proper formatted file.\
                        / GPU unavailable. Try another time.'
            #print("Saving..")
            df.to_csv('./download/result'+str(i)+'.csv',index=False,header=True)
            #print("saved")
            location = '/dash/urlToDownload?value={}'.format('result'+str(i)+'.csv')
            i+=1
            refslist.append(html.Li(html.A(fname, href=location)))
    return refslist

@app.callback(
    Output('input_text_button','n_clicks'),
    [Input('reset_button','n_clicks')]
)
def reset(clicks):
    return 0


if __name__ == '__main__':
    app.run_server(debug=False)