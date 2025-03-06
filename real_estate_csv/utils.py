import plotly.io as pio
pio.renderers.default = "notebook"


import plotly.express as px
import json
import requests
import pandas as pd


def create_city_mapping(X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame)

    X['unique_city_id'] = X.apply(lambda row: (row['departement'], row['id_ville']), axis=1)

    # Creating a city mapping to re-use for the predictions and saving it to .csv
    city_mapping = X[['unique_city_id', 'ville']]

    return city_mapping


def choropleth_map(df):

    # Load GeoJSON file using requests
    geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    response = requests.get(geojson_url)
    france_geojson = response.json()


    # Create the choropleth map
    fig = px.choropleth(
        df,
        geojson=france_geojson,
        locations="departement",  # Match department codes in DataFrame
        featureidkey="properties.code",  # Match department codes in GeoJSON
        # color="value",  # Column to determine color
        hover_name="ville",  # Column to display on hover
        hover_data=["prix"]  # Additional info on hover
    )

    # Update layout for better visualization
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title="Interactive Map of French Departments")

    # Show the map
    # fig.show()
    # # fig.write_html("map.html")
    # fig.show(renderer="browser")
