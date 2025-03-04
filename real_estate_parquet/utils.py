import plotly.io as pio
pio.renderers.default = "notebook"


import plotly.express as px
import json
import requests
import pandas as pd


def create_city_mapping(X: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a mapping of unique city identifiers by combining 'departement' and 'id_ville' columns.

    Specifically, it:
    - Selects the 'departement', 'id_ville', and 'ville' columns from the input DataFrame.
    - Constructs a 'unique_city_id' by combining 'departement' and 'id_ville' into a tuple string.
    - Drops the 'departement' and 'id_ville' columns from the resulting DataFrame.

    Args:
        X (pd.DataFrame): The input DataFrame containing the 'departement', 'id_ville', and 'ville' columns.

    Returns:
        pd.DataFrame: A DataFrame containing the 'ville' and 'unique_city_id' columns, where 'unique_city_id' is the combination of 'departement' and 'id_ville'.
    """

    assert isinstance(X, pd.DataFrame)

    city_mapping = X[['departement', 'id_ville', 'ville']].copy()
    city_mapping['unique_city_id'] = city_mapping.apply(lambda row: f"({row['departement']},{row['id_ville']})", axis=1)
    city_mapping.drop(columns=['departement', 'id_ville'], axis=1, inplace=True)

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
