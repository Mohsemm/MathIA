import math
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import pandas as pd

# --- CONFIGURATION ---
EXPORT_TO_EXCEL = False  # Set to True to export data to Excel file

# --- 1. Define the Distance Calculation Models ---
R = 6371.0  # Earth's mean radius in km

def haversin(theta):
    """Calculates the haversine of an angle given in radians, as per the IA derivation."""
    return math.sin(theta / 2)**2

def planar_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance using the Planar/Equirectangular model (Model A)."""
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    if delta_lon > 180: delta_lon -= 360
    elif delta_lon < -180: delta_lon += 360
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lon_rad = math.radians(delta_lon)
    midpoint_lat_rad = math.radians((lat1 + lat2) / 2)
    x = delta_lon_rad * math.cos(midpoint_lat_rad)
    y = math.radians(delta_lat)
    return R * math.sqrt(x**2 + y**2)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance using the Haversine formula (Model B)."""
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    if delta_lon > math.pi: delta_lon -= 2 * math.pi
    elif delta_lon < -math.pi: delta_lon += 2 * math.pi
    haversin_alpha = haversin(delta_lat) + math.cos(lat1_rad) * math.cos(lat2_rad) * haversin(delta_lon)
    alpha = 2 * math.asin(math.sqrt(haversin_alpha))
    return R * alpha

def get_route_characteristics(lat1, lon1, lat2, lon2):
    """Calculates initial bearing and max latitude for a route."""
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    delta_lon = lon2_rad - lon1_rad
    if delta_lon > math.pi: delta_lon -= 2 * math.pi
    elif delta_lon < -math.pi: delta_lon += 2 * math.pi
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    initial_bearing_rad = math.atan2(y, x)
    max_lat_rad = math.acos(abs(math.sin(initial_bearing_rad) * math.cos(lat1_rad)))
    return math.degrees(max_lat_rad)

# --- 2. Balanced Input Data ---
routes = [
    # Your 4 original routes are integrated
    {'name': 'Dubrovnik -> Milan', 'start': (42.559124, 18.267439), 'end': (45.666813, 9.698662)},
    {'name': 'Milan -> Casablanca', 'start': (45.6301, 8.714771), 'end': (33.329132, -7.578973)},
    {'name': 'Dubai -> Dubrovnik', 'start': (25.251686, 55.37014), 'end': (42.559124, 18.267439)},
    {'name': 'Casablanca -> Dubai', 'start': (33.394318, -7.60144), 'end': (25.251686, 55.37014)},
    
    # Additional balanced routes
    {'name': 'Amsterdam -> Brussels', 'start': (52.295151, 4.762096), 'end': (50.902508, 4.456756)},
    {'name': 'LA -> Tokyo', 'start': (33.946953, -118.406807), 'end': (35.549389, 139.769028)},
    {"name": "London -> Singapore", "start": (51.477500, -0.461390), "end": (1.359170, 103.989440)},
    {"name": "Cairo -> Johannesburg", "start": (30.121940, 31.405560), "end": (-26.133610, 28.242220)},
    {"name": "Tromsø -> Alta", "start": (69.681390, 18.917780), "end": (69.976110, 23.371670)},
    {"name": "Oslo -> Helsinki", "start": (60.202780, 11.083890), "end": (60.317220, 24.963330)},
    {"name": "Helsinki -> Svalbard", "start": (60.317220, 24.963330), "end": (78.246110, 15.465560)},
    {"name": "Winnipeg -> Reykjavik", "start": (49.910000, -97.240000), "end": (63.985000, -22.605556)},
    {"name": "Anchorage -> Reykjavik", "start": (61.174170, -149.998330), "end": (63.985000, -22.605556)},
    {"name": "Seattle -> Helsinki", "start": (47.448890, -122.309440), "end": (60.317220, 24.963330)},
    {"name": "Santiago -> Sydney", "start": (-33.392780, -70.785560), "end": (-33.946110, 151.177220)},
    {"name": "Chicago -> Hong Kong", "start": (41.978610, -87.904720), "end": (22.308000, 113.918500)},
]

# --- 3. Main Calculation and Categorization Loop ---
results = []
for route in routes:
    lat1, lon1 = route['start']
    lat2, lon2 = route['end']
    max_lat = get_route_characteristics(lat1, lon1, lat2, lon2)
    category = 'high' if max_lat > 60 else 'normal'
    dist_model_a = planar_distance(lat1, lon1, lat2, lon2)
    dist_model_b = haversine_distance(lat1, lon1, lat2, lon2)
    dist_benchmark = geodesic((lat1, lon1), (lat2, lon2)).kilometers
    percent_error_a = abs((dist_model_a - dist_benchmark) / dist_benchmark) * 100
    percent_error_b = abs((dist_model_b - dist_benchmark) / dist_benchmark) * 100
    results.append({
        "distance": dist_benchmark, "accuracy_a": 100 - percent_error_a,
        "accuracy_b": 100 - percent_error_b, "category": category
    })

# --- 4. Export Data to Excel ---
def export_to_excel(all_results, filename='math_ia_regression_data.xlsx'):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        normal_routes = sorted([r for r in all_results if r['category'] == 'normal'], key=lambda x: x['distance'])
        high_routes = sorted([r for r in all_results if r['category'] == 'high'], key=lambda x: x['distance'])
        df_normal = pd.DataFrame(normal_routes)[['distance', 'accuracy_a', 'accuracy_b']]
        df_normal.columns = ['Benchmark Distance (x)', 'Model A Accuracy (y)', 'Model B Accuracy (y)']
        df_normal.to_excel(writer, sheet_name='Normal Latitude Routes', index=False)
        df_high = pd.DataFrame(high_routes)[['distance', 'accuracy_a', 'accuracy_b']]
        df_high.columns = ['Benchmark Distance (x)', 'Model A Accuracy (y)', 'Model B Accuracy (y)']
        df_high.to_excel(writer, sheet_name='High Latitude Routes', index=False)
    print(f"\nData successfully exported to '{filename}'\n")

# --- 5. Plotting Function ---
def create_plot(data, title, model_a_fit_func, model_b_fit_func):
    if not data: return
    data.sort(key=lambda x: x['distance'])
    
    distances = np.array([r['distance'] for r in data])
    acc_a = np.array([r['accuracy_a'] for r in data])
    acc_b = np.array([r['accuracy_b'] for r in data])

    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(distances, acc_a, color='red')
    ax.scatter(distances, acc_b, color='blue')

    x_fit = np.linspace(0, distances.max(), 400)
    y_fit_a = np.minimum(model_a_fit_func(x_fit), 100)
    y_fit_b = np.minimum(model_b_fit_func(x_fit), 100)
    
    ax.plot(x_fit, y_fit_a, color='red', linestyle='-', linewidth=2, label='Model A (Planar)')
    ax.plot(x_fit, y_fit_b, color='blue', linestyle='-', linewidth=2, label='Model B (Haversine)')

    y_min, y_max = 70, 100.5
    x_major_ticks = 2000
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Benchmark Distance (km)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, distances.max() * 1.05)
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_major_ticks))

# --- 6. DEFINE YOUR REGRESSION FUNCTIONS HERE ---

def normal_model_a_fit(x):
    # nA(x) = 99.41274 + 0.0004126016x - 1.041684*10^(-7)x^2
    return 99.41274 + 0.0004126016*x - (1.041684*10**-7)*x**2

def normal_model_b_fit(x):
    # nB(X) = 0.000002287517x + 99.8551
    return 0.000002287517*x + 99.8551

def high_model_a_fit(x):
    # hA(x) = 100.1637 + 0.0002421034x - 6.527577*10^(-7)x^2 + 4.226957*10^(-11)x^3
    return 100.1637 + 0.0002421034*x - (6.527577*10**-7)*x**2 + (4.226957*10**-11)*x**3

def high_model_b_fit(x):
    # hB(x) = 0.000006119725x + 99.66944
    return 0.000006119725*x + 99.66944

# --- 7. Export Data and Generate Graphs ---
if EXPORT_TO_EXCEL:
    export_to_excel(results)
else:
    print("Excel export skipped (EXPORT_TO_EXCEL = False)")

normal_routes_data = [r for r in results if r['category'] == 'normal']
high_routes_data = [r for r in results if r['category'] == 'high']

create_plot(normal_routes_data, 'Model Accuracy on Normal Latitude Routes (φ_max < 60°)', normal_model_a_fit, normal_model_b_fit)
create_plot(high_routes_data, 'Model Accuracy on High Latitude Routes (φ_max > 60°)', high_model_a_fit, high_model_b_fit)

plt.show()