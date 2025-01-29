import plotly.io as pio
pio.renderers.default = "notebook"


import plotly.express as px
import json
import requests
import pandas as pd

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

"""----------------------------------------------------------------------------"""

# Function to expand yearly debt ratio into monthly data
def expand_to_monthly_evolution(df):
    monthly_data = []
    for i in range(len(df) - 1):
        # Get the current year and next year values
        start_year = df.index[i].year
        start_value = df.iloc[i]['taux_endettement']
        end_value = df.iloc[i + 1]['taux_endettement']

        # Calculate the monthly change
        monthly_change = (end_value - start_value) / 12

        # Generate monthly dates for the current year
        months = pd.date_range(start=f"{start_year}-01-01", end=f"{start_year}-12-31", freq='MS')

        # Create monthly debt ratios
        monthly_ratios = [start_value + (monthly_change * m) for m in range(12)]

        # Append the monthly data for this year
        monthly_data.append(pd.DataFrame({'date': months, 'taux_endettement': monthly_ratios}).set_index('date'))

    # Concatenate all rows into a single DataFrame
    return pd.concat(monthly_data)

# # Apply the function to expand the data
# taux_endettement_mensuel_df = expand_to_monthly_evolution(taux_endettement_df)

"""----------------------------------------------------------------------------"""


def create_city_mapping(X: pd.DataFrame, cache_path='city_mapping.csv') -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame)

    X['unique_city_id'] = X.apply(lambda row: (row['departement'], row['id_ville']), axis=1)

    # Creating a city mapping to re-use for the predictions and saving it to .csv
    city_mapping = X[['unique_city_id', 'ville']].drop_duplicates()

    return city_mapping





    # xgb_model = xgboost_model(
    #                         n_estimators=1000,           # Number of trees (boosting rounds)
    #                         learning_rate=0.01,          # Step size shrinkage
    #                         max_depth=6,                 # Maximum depth of trees
    #                         min_child_weight=1,          # Minimum sum of instance weights needed in a child
    #                         subsample=0.8,               # Fraction of samples for each tree
    #                         colsample_bytree=0.8,        # Fraction of features for each tree
    #                         gamma=0,                     # Minimum loss reduction to make a split
    #                         reg_alpha=0.1,               # L1 regularization
    #                         reg_lambda=1.0,              # L2 regularization
    #                         objective='reg:squarederror', # Objective function for regression
    #                         random_state=42,             # For reproducibility
    #                         verbosity=1                  # Verbosity of output during training
    #                         )
