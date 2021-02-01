import dash
import dash_auth

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server

auth = dash_auth.BasicAuth(
    app,
    {'admin': 'admin',
     'admin2': 'admin2'})
