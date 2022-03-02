"""
ipyleaflet map for validating candidate TPA sites
"""

import geopandas as gpd
import ipyleaflet as ipyl
import ipywidgets as ipyw


# define XYZ tile layers to add to map
tile_layers = {
    "google": "http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "planet": "https://tiles.planet.com/basemaps/v1/planet-tiles/global_monthly_2020_06_mosaic/gmap/{z}/{x}/{y}.png?api_key=73112b15212f4e7bb15e35f1b144f049"
}

class Validator:
    def __init__(self, tpa_in, tpa_out):
        # read input data
        self.gdf = self.read_geojson(tpa_in)
        print(tile_layers['planet'])

        # build map
        self.map = self.create_map()

        # add base controls
        self.add_base_controls()

        # add basemap selector
        self.add_basemap_selector()

        # add feature selector
        self.add_feature_selector()

        # add save button
        self.tpa_out = tpa_out
        self.add_save_button()


    def create_map(self):
        # create basic map
        map_kwargs = dict()
        map_kwargs['dragging'] = True
        map_kwargs['zoom_control'] = False
        map_kwargs['attribution_control'] = False
        map_kwargs['scroll_wheel_zoom'] = True
        map_kwargs['zoom'] = 18
        map_kwargs['center'] = [-7.455435243204367, 111.80284213645794]
        m = ipyl.Map(**map_kwargs)
        m.layout = ipyw.Layout(height='800px')

        # add google layer as default
        layer = ipyl.TileLayer(url=tile_layers['google'],
                               name='google')
        m.add_layer(layer)

        return m


    def add_base_controls(self):
        zc = ipyl.ZoomControl(position='topright')
        self.map.add_control(zc)

        fsc = ipyl.FullScreenControl(position='topright')
        self.map.add_control(fsc)

        gpw_file = open('gpw.png', 'rb')
        gpw = ipyw.Image(
            value=gpw_file.read(),
            format='png'
        )
        gpw.layout.object_fit = 'cover'
        wc = ipyl.WidgetControl(widget=gpw, position='topleft')
        self.map.add_control(wc)


    def add_basemap_selector(self):
        basemaps = tile_layers.keys()
        layout = ipyw.Layout(width='35px', height='35px')
        basemap_selector = ipyw.Dropdown(
            options=basemaps,
            value=list(basemaps)[0],
            description='basemap',
            continuous_update=False
        )

        def interact_basemap(change):
            old_basemap = self.map.layers[0]
            self.map.remove_layer(old_basemap)
            
            value = change['new']
            new_layer = ipyl.TileLayer(url=tile_layers[value],
                                       name=value)
            self.map.add_layer(new_layer)

        basemap_selector.observe(interact_basemap, names='value')

        control = ipyl.WidgetControl(widget=basemap_selector, position='bottomright')
        self.map.add_control(control)


    def add_save_button(self):
        save_button = ipyw.ToggleButton(
            value=True,
            description='Save work',
            disabled=False,
            button_style='info',
            icon='bookmark'
        )

        def click_save(change):
            self.gdf.to_file(self.tpa_out, driver='GeoJSON')

        save_button.observe(click_save, names='value')

        control = ipyl.WidgetControl(widget=save_button, position='bottomleft')
        self.map.add_control(control)


    def add_feature_selector(self):
        dump_selector = ipyw.RadioButtons(
            options=['yes', 'no', 'maybe'],
            value=None,
            description='Dump site?',
            disabled=False
        )

        feature_progress = ipyw.IntProgress(
            description="Progress: ",
            value=0,
            min=0,
            max=len(self.gdf),
            disabled=False,
            orientation='horizontal',
            bar_style='info'
        )

        prev_button = ipyw.ToggleButton(
            value=True,
            description='',
            disabled=False,
            button_style='success',
            icon='arrow-left',
        )

        next_button = ipyw.ToggleButton(
            value=True,
            description='',
            disabled=False,
            button_style='success',
            icon='arrow-right'
        )

        button_pane = ipyw.HBox(
            [
                prev_button,
                next_button
            ]
        )

        feature_pane = ipyw.VBox(
            [
                feature_progress,
                dump_selector,
                button_pane
            ]
        )
        feature_pane.layout.display = 'flex'
        feature_pane.layout.align_items = 'stretch'

        control = ipyl.WidgetControl(widget=feature_pane, position='bottomleft')
        self.map.add_control(control)

        def update_feature():
            self.map.center = (self.gdf.iloc[self.fctr]['lat'], self.gdf.iloc[self.fctr]['lon'])
            marker = ipyl.CircleMarker(location=(self.gdf.iloc[self.fctr]['lat'], self.gdf.iloc[self.fctr]['lon']),
                radius=25, 
                color='#ffce00', 
                fill_opacity=0,
                weight=3)
            self.map.add_layer(marker)
            feature_progress.value = self.fctr
            
            for option in dump_selector.options:
                if self.gdf.iloc[self.fctr][option] == True:
                    dump_selector.value = option

        def click_next(change):
            if self.fctr < len(self.gdf):
                self.fctr += 1
                dump_selector.value = None
                update_feature()

        def click_prev(change):
            if self.fctr > 0:
                self.fctr -= 1
                dump_selector.value = None
                update_feature()

        def click_dump(change):
            if change['new'] is not None:
                self.gdf.at[self.fctr, change['new']] = True
                for option in dump_selector.options:
                    if option != change['new']:
                        self.gdf.at[self.fctr, option] = False

        self.fctr = 0
        update_feature()
        prev_button.observe(click_prev, names='value')
        next_button.observe(click_next, names='value')
        dump_selector.observe(click_dump, names='value')
        

    def read_geojson(self, input_file):
        gdf = gpd.read_file(input_file)
        
        # add boolean flags
        gdf["yes"] = False
        gdf["no"] = False
        gdf["maybe"] = False

        return gdf


