import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from calendar import month_name
import random
import numpy as np

from owslib.wms import WebMapService
from rasterio.io import MemoryFile

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#+++++ Utility functions
def load_data():
    df = pd.read_excel('./NSIDC_Regional_Daily_Data.xlsx', sheet_name=None)
    sheet_names = [sheet for sheet in df.keys() if variable in sheet]
    for sheet in sheet_names:
        df[sheet]['month'] = df[sheet]['month'].ffill()
    return df, sheet_names

def get_color_hashmap():
    colors = px.colors.cyclical.IceFire
    return {
        'January': colors[2], 'February': colors[3], 'March': colors[4],
        'April': colors[7], 'May': colors[9], 'June': colors[10], 'July': colors[11],
        'August': colors[12], 'September': colors[13], 'October': colors[14],
        'November': colors[15], 'December': colors[0]
    }

def gen_random_linestyle():
    return f'{random.choice(range(0,10,3)):.0f}px {random.choice(range(1,10,3))}px {random.choice(range(1,10,2))}px'


def retrieve_seaice_raster(month_int, year):
    # WMS URL for NSIDC
    wms_url = 'https://nsidc.org/api/mapservices/NSIDC/wms'
    wms = WebMapService(wms_url, version='1.1.1')
    
    # Define request parameters
    layer_name = 'NSIDC:g02135_extent_raster_monthly_n'
    bbox = (-4100000.0, -2600000.0, 3100000.0, 3100000.0)  # Bounding box lower left x/y, upper right x/y
    crs = 'EPSG:3411'  # Polar stereographic projection
    width, height = 500, 500  # Image dimensions
    styles = 'g02135_extent_raster_basemap'

    time = f'{year}-{month_int:02d}'
    
    # Request GeoTIFF from WMS
    response = wms.getmap(
        layers=[layer_name],
        bbox=bbox,
        srs=crs,
        format='image/geotiff',
        size=(width, height),
        styles=[styles],
        time=time
    )

    #we can find grid for plotting ahead of time
    x = np.linspace(bbox[0], bbox[2], width)
    y = np.linspace(bbox[3], bbox[1], height)
    xx, yy = np.meshgrid(x,y)

    # Read GeoTIFF data into rasterio with MemoryFile
    with MemoryFile(response) as memfile:
        with memfile.open() as dataset:
            # Read the raster data into a numpy array
            data = dataset.read(1, masked=False)     
            
    return xx, yy, data
#+++

#+++Read in data
variable = 'Extent-km^2'
NSIDC_yrs=range(1979,2025)


# Create NSIDC Data and Mappings
df, sheet_names = load_data()
region_names = [x.replace(f'-Extent-km^2', "") for x in sheet_names]
color_hashmap = get_color_hashmap()
linestyle_hashmap = {sheet: gen_random_linestyle() for sheet in sheet_names}
###


#+++Plotting functions
# Timeseries plots
def create_timeseries_plot(selected_data, hoverData):
    if not selected_data:
        return go.Figure(), []

    fig = go.Figure()
    trace_info = []
    for region in selected_data.keys():
        for month in selected_data[region]:
            data = df[region]
            df_month = data[data['month'] == month]
            NSIDC_yrs = df_month.columns[2:]
            df_month_avg = df_month[NSIDC_yrs].mean(axis=0)

            fig.add_trace(go.Scatter(
                x=NSIDC_yrs, y=df_month_avg,
                mode='markers+lines',
                line=dict(color=color_hashmap[month], dash=linestyle_hashmap[region]),
                marker=dict(color='black', size=8, line=dict(color='black', width=1)),
                name=f"{region.replace(f'-Extent-km^2', '')} ({month})"
            ))

            trace_info.append((region, month))

    if hoverData:
        fig = add_inset_plot(fig, hoverData, trace_info)

    return fig, trace_info

# Inset plot with daily data
def add_inset_plot(fig, hoverData, trace_info):
    clicked_point = hoverData['points'][0]
    clicked_year = clicked_point['x']
    clicked_curve = clicked_point['curveNumber']
    clicked_value = clicked_point['y']
    clicked_reg, clicked_mon = trace_info[clicked_curve]

    df_daily = df[clicked_reg]
    df_daily = df_daily[df_daily['month'] == clicked_mon]
    df_daily = df_daily[NSIDC_yrs]

    # Add traces for the clicked year’s daily sea ice evolution
    fig.add_trace(go.Scatter(
        x=np.array(range(len(df_daily))),
        y=df_daily[clicked_year].values,
        mode='markers+lines',
        line=dict(color=color_hashmap[clicked_mon], dash=linestyle_hashmap[clicked_reg]),
        marker=dict(color='black', size=2, line=dict(color='black', width=1)),
        name=f"{clicked_reg.replace(f'-{variable}', '')} ({clicked_mon})",
        xaxis='x2', yaxis='y2'
    ))

    mu_evolution = df_daily.mean(axis=1).values
    st_evolution = df_daily.std(axis=1).values

    fig.add_trace(go.Scatter(x=np.array(range(len(df_daily))),
                             y=mu_evolution, mode='lines', line=dict(color='black'),
                             xaxis='x2', yaxis='y2'))

    fig.add_trace(go.Scatter(x=np.array(range(len(df_daily))),
                             y=mu_evolution + st_evolution, mode='lines', line=dict(color='red'),
                             xaxis='x2', yaxis='y2'))

    fig.add_trace(go.Scatter(x=np.array(range(len(df_daily))),
                             y=mu_evolution - st_evolution, mode='lines', line=dict(color='red'),
                             fill='tonexty', fillcolor='rgba(128,128,128,0.25)',
                             xaxis='x2', yaxis='y2'))

    return fig

#sea ice area map
def plot_seaice_raster(y1, m1, y2, m2):

    m1 = list(calendar.month_abbr).index(m1)
    m2 = list(calendar.month_abbr).index(m2)

    print(m1, m2)

    #retrieve raster data
    xx,yy, seaice_max = retrieve_seaice_raster(m1, y1)
    _,_,   seaice_min = retrieve_seaice_raster(m2, y2)
                     
    # Set up masked arrays for each condition
    no_ice =       np.ma.masked_where(~ ((seaice_max == 0) & (seaice_min == 0)) , seaice_max)
    year_min_ice = np.ma.masked_where(~ ((seaice_max == 0) & (seaice_min == 1)) , seaice_min)
    year_max_ice = np.ma.masked_where(~ ((seaice_max == 1) & (seaice_min == 0)) , seaice_max)
    both_ice =     np.ma.masked_where(~ ((seaice_max == 1) & (seaice_min == 1)) , seaice_max)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

    
    cmap = plt.cm.get_cmap('Blues_r')
    colors = [cmap(0.), cmap(0.25), 'w', cmap(0.75)]
    labels = ['No ice', f'{y1} ice', 'Both year ice', f'{y2} ice'] # Define colors for each class
    
    ax_seaice.pcolormesh(xx, yy, no_ice,       cmap=mcolors.ListedColormap([colors[0]]), transform=ccrs.epsg(3411), zorder=1)
    ax_seaice.pcolormesh(xx, yy, year_min_ice, cmap=mcolors.ListedColormap([colors[1]]), transform=ccrs.epsg(3411), zorder=5)
    ax_seaice.pcolormesh(xx, yy, year_max_ice, cmap=mcolors.ListedColormap([colors[3]]), transform=ccrs.epsg(3411), zorder=10)
    ax_seaice.pcolormesh(xx, yy, both_ice,     cmap=mcolors.ListedColormap([colors[2]]), transform=ccrs.epsg(3411), zorder=15)

    ax_seaice.add_feature(cfeature.LAND, zorder=20,color='gray') #add land
    
    # Add a legend
    handles = [plt.Line2D([0], [0], marker='s', color='k', markerfacecolor=c, markersize=10) for c in colors]
    ax_seaice.legend(handles, labels, loc='lower right').set_zorder(200)
    #+++

    return fig 

import io
import base64

def fig_to_base64(fig):
    # Convert the Matplotlib figure to a PNG image
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    
    # Encode the image as base64
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    return f'data:image/png;base64,{encoded_image}'
#+++

#+++ Dash app layout
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Dropdown(
        id='region-dropdown',
        options=[{'label': reg_name, 'value': sheet_name} for reg_name, sheet_name in zip(region_names, sheet_names)],
        value=sheet_names[0],
        multi=False
    ),
    html.Div(id='checklist-container'),
    html.Button('Clear Plot', id='clear-button'),
    dcc.Store(id='stored-month-selections', data={}),
    dcc.Store(id='sea_ice_map_selection1', data={}),
    dcc.Store(id='sea_ice_map_selection2', data={}),
    dcc.Store(id='toggle-state', data=True),
    dcc.Graph(id='plot-yearly_regi_moni'),
    dcc.Graph(id='area-plots')
])
#+++


#++++ Dash updates/callbacks
# 
@app.callback(
    Output('checklist-container', 'children'),
    [Input('region-dropdown', 'value'), Input('stored-month-selections', 'data')]
)
def init_region_months(dropdown_region, saved_months):
    saved_checks = saved_months.get(dropdown_region, [])
    return dcc.Checklist(
        id='month-checklist',
        options=[{'label': month, 'value': month} for month in color_hashmap.keys()],
        value=saved_checks,
        inline=True
    )

#
@app.callback(
    Output('stored-month-selections', 'data'),
    [Input('month-checklist', 'value')],
    [State('region-dropdown', 'value'), State('stored-month-selections', 'data')],
    Input('clear-button', 'n_clicks'),
)
def update_region_months(selected_months, dropdown_region, saved_months, n_clicks):
    if not selected_months:
        saved_months.pop(dropdown_region, None)
    else:
        saved_months[dropdown_region] = selected_months

    #\\TODO: fix this, can't plot after clicking
    #if n_clicks:
    #    saved_months = {}

    return saved_months

#this updates the two main plots
@app.callback(
    Output('plot-yearly_regi_moni', 'figure'),
    Input('stored-month-selections', 'data'),
    Input('plot-yearly_regi_moni', 'hoverData'),
    State('sea_ice_map_selection1', 'data'),
    State('sea_ice_map_selection2', 'data'),
)
def update_plot(selected_data, hoverData, raster_select1, raster_select2):
    fig, trace_info = create_timeseries_plot(selected_data, hoverData)

    # Set layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Mean Sea Ice Extent (km^2)',
        showlegend=True,
        xaxis2=dict(domain=[0.65, 0.95], title='Day of Month', title_standoff=10, position=0.95, anchor='y2'),
        yaxis2=dict(domain=[0.65, 0.95], title='Sea Ice Extent (km^2)', title_standoff=10, position=0.65, anchor='x2'),
        width=1800,
        height=900,
    )
    
    # img_src = None
    # Add code to check if raster_select1 and raster_select2 are valid
    if raster_select1 and raster_select2:
        print(f'raster_select1: {raster_select1}')
        print(f'raster_select2: {raster_select2}')
    #     print(raster_select1)
    #     y1 = raster_select1['x'] #this gets year
    #     m1 = trace_info[raster_select1['curve_num']][1] #this gets month

    #     y2 = raster_select2['x'] #this gets year
    #     m2 = trace_info[raster_select2['curve_num']][1] #this gets month

    #     fig2=plot_seaice_raster(y1,m1,y2,m2)

    #     img_src = fig_to_base64(fig2)

    return fig

#this stores ddata for the area extent data based on clicks
@app.callback(
    Output('sea_ice_map_selection1', 'data'),
    Output('sea_ice_map_selection2', 'data'),
    Output('toggle-state', 'data'),
    Input('plot-yearly_regi_moni', 'clickData'),
    State('sea_ice_map_selection1', 'data'),
    State('sea_ice_map_selection2', 'data'),
    State('toggle-state', 'data'),
    )
def store_seaice_click(clickData, selection1, selection2, toggle_state):
    if clickData:
        # Extract click information (e.g., x, y values)
        curve_num = clickData['points'][0]['curveNumber']
        clicked_x = clickData['points'][0]['x']
        clicked_y = clickData['points'][0]['y']
        
        click_point = {'curve': curve_num, 'x': clicked_x, 'y': clicked_y}

        if toggle_state: selection1=click_point
        else: selection2=click_point

        toggle_state = not toggle_state

        
    return selection1, selection2, toggle_state

if __name__ == '__main__':
    app.run_server(debug=True)
