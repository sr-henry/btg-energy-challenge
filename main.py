import os
import re
import time
from functools import wraps
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


def read_data_file(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        raw_file = f.readlines()

    list_dados = [line.split() for line in raw_file]
    float_raw_lines = [list(map(float, raw_line)) for raw_line in list_dados]
    return pd.DataFrame(float_raw_lines, columns=['lat', 'long', 'data_value'])


def read_contour_file(file_path: str) -> pd.DataFrame:
    line_split_comp = re.compile(r'\s*,')

    with open(file_path, 'r') as f:
        raw_file = f.readlines()

    l_raw_lines = [line_split_comp.split(raw_file_line.strip()) for raw_file_line in raw_file]
    l_raw_lines = list(filter(lambda item: bool(item[0]), l_raw_lines))
    float_raw_lines = [list(map(float, raw_line))[:2] for raw_line in l_raw_lines]
    header_line = float_raw_lines.pop(0)
    assert len(float_raw_lines) == int(header_line[0])
    return pd.DataFrame(float_raw_lines, columns=['lat', 'long'])


def calculate_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Function "{func.__name__}" executed in {execution_time:.6f}s.')
        return result
    return wrapper


@calculate_execution_time
def apply_contour(contour_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters points in data_df that lie within the region defined by contour_df.

    Parameters:
        contour_df (pd.DataFrame): The dataframe containing the region's latitude and longitude.
        data_df (pd.DataFrame): The dataframe containing points to be filtered.

    Returns:
        pd.DataFrame: The filtered dataframe containing points within the region.
    """
    region_polygon: Polygon = Polygon(contour_df[['lat', 'long']].values)

    def point_within_region(point: Point, region_polygon: Polygon) -> bool:
        return region_polygon.contains(Point(point['lat'], point['long']))
    
    filtered_data_df = data_df[data_df.apply(point_within_region, region_polygon=region_polygon, axis=1)]

    return filtered_data_df


def get_forcast_dates(forcast_file_name: str) -> list:
    matches = re.findall(r'(\d{2})(\d{2})(\d{2})', forcast_file_name)
    dates_ddmmyy = [f'{day}/{month}/{year}' for day, month, year in matches]
    return dates_ddmmyy


def read_data_files(forcast_folder: str) -> pd.DataFrame:
    combined_data: pd.DataFrame = pd.DataFrame()

    for forcast_file in os.listdir(forcast_folder):
        date0, date1 = get_forcast_dates(forcast_file)
       
        df = read_data_file(f'{forcast_folder}/{forcast_file}')
        df = df.assign(forecast_date=date0, forecasted_date=date1)
        
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    return combined_data


def map_plot(contour_df: pd.DataFrame, filtered_data_df: pd.DataFrame, ax) -> None:
    points_to_plot = filtered_data_df[['lat', 'long']].drop_duplicates()
    points_to_plot['side'] = 0.4

    ax.set_title('Camargos - Bacia do Grande')

    ax.plot(contour_df['lat'], contour_df['long'], c='#0a0a0a', label='Region')
    
    for _, row in points_to_plot.iterrows():
        center_x, center_y, side_length = row['lat'], row['long'], row['side']

        x_left = center_x - side_length / 2
        y_bottom = center_y - side_length / 2

        ax.add_patch(
            plt.Rectangle((x_left, y_bottom), side_length, side_length, fill=True, facecolor='#00d9ff',  edgecolor='#ff0000')
        )
    
    ax.scatter(points_to_plot['lat'], points_to_plot['long'], c='g', label='Points within Region')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.axis('equal')
    ax.legend()


def forcast_plot(forcast_df: pd.DataFrame, ax) -> None:
    
    forcast_df.plot(kind='barh', ax=ax)
    
    ax.set_title('Preciptation Forcast ISO x ACC')
    ax.legend(['Forcast Precipitation Iso','Forcast Precipitation Acc'])
    
    for container in ax.containers:
        ax.bar_label(container)


def result_plot(contour_df: pd.DataFrame, filtered_data_df: pd.DataFrame, forcast_df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    map_plot(contour_df=contour_df, filtered_data_df=filtered_data_df, ax=ax1)
    forcast_plot(forcast_df=forcast_df, ax=ax2)
    plt.tight_layout()
    plt.savefig('main_result.png')
    plt.show()


def main() -> None:
    contour_df: pd.DataFrame = read_contour_file('PSATCMG_CAMARGOS.bln')
    data_df: pd.DataFrame = read_data_files('forecast_files')
    filtered_data_df: pd.DataFrame = apply_contour(contour_df=contour_df, data_df=data_df)

    # Considerando que a medida utilizada para o valor da precipitaçõa do modelo esteja em unidades de comprimento de água
    # Temos que fazer a média (`mean`) das sub regiões, mas caso o valor fosse em volume de água seria uma soma (`sum`)
    forcast_df = filtered_data_df.groupby(by=['forecast_date','forecasted_date'])[['data_value']].mean()
    forcast_df['data_value_acc'] = forcast_df['data_value'].expanding(1).sum()

    result_plot(contour_df, filtered_data_df, forcast_df)

if __name__ == '__main__':
    main()
