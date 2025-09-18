import math
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

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
    {'name': 'Amsterdam -> Brussels', 'start': (52.31, 4.76), 'end': (50.90, 4.54)},
    {'name': 'LA -> Tokyo', 'start': (33.9425, -118.4081), 'end': (35.7647, 139.7867)},
    {'name': 'London -> Singapore', 'start': (51.4700, -0.4543), 'end': (1.3644, 103.9915)},
    {'name': 'Cairo -> Johannesburg', 'start': (30.1219, 31.4056), 'end': (-26.1392, 28.2460)},
    {'name': 'Tromsø -> Alta', 'start': (69.6819, 18.9169), 'end': (69.9769, 23.3717)},
    {'name': 'Oslo -> Helsinki', 'start': (60.1976, 11.1004), 'end': (60.3172, 24.9633)},
    {'name': 'Helsinki -> Svalbard', 'start': (60.3172, 24.9633), 'end': (78.2461, 15.4656)},
    {'name': 'Winnipeg -> Reykjavik', 'start': (49.9100, -97.2400), 'end': (63.9850, -22.6056)},
    {'name': 'Anchorage -> Reykjavik', 'start': (61.1743, -149.9963), 'end': (63.9850, -22.6056)},
    {'name': 'Seattle -> Helsinki', 'start': (47.4490, -122.3093), 'end': (60.3172, 24.9633)},
    {'name': 'Santiago -> Sydney', 'start': (-33.3930, -70.7858), 'end': (-33.9461, 151.1772)},
    {'name': 'Chicago -> Hong Kong', 'start': (41.9742, -87.9073), 'end': (22.3080, 113.9185)},
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

# --- 4. Plotting Function with YOUR Regression Formulas ---
def create_plot(data, title, model_a_fit_func, model_b_fit_func):
    if not data: return
    data.sort(key=lambda x: x['distance'])
    
    distances = np.array([r['distance'] for r in data])
    acc_a = np.array([r['accuracy_a'] for r in data])
    acc_b = np.array([r['accuracy_b'] for r in data])

    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(distances, acc_a, color='red')
    ax.scatter(distances, acc_b, color='blue')

    # Generate points for the smooth best-fit lines using YOUR formulas
    x_fit = np.linspace(0, distances.max(), 400)
    y_fit_a = np.minimum(model_a_fit_func(x_fit), 100) # Cap line at 100%
    y_fit_b = np.minimum(model_b_fit_func(x_fit), 100) # Cap line at 100%
    
    ax.plot(x_fit, y_fit_a, color='red', linestyle='-', linewidth=2, label='Model A (Planar)')
    ax.plot(x_fit, y_fit_b, color='blue', linestyle='-', linewidth=2, label='Model B (Haversine)')

    # --- Customizable Plot Limits & Ticks ---
    y_min, y_max = 75, 100.5
    x_major_ticks = 2000
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Benchmark Distance (km)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, distances.max() * 1.05)
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_major_ticks))

# --- 5. Define YOUR Regression Functions ---
def normal_model_a_fit(x):
    return 99.42339 + 0.0004061479*x - 1.037691e-7*x**2

def normal_model_b_fit(x):
    return 0.000002277715*x + 99.85508

def high_model_a_fit(x):
    return (4.3731e-11)*x**3 - (6.83021e-7)*x**2 + (0.0004136)*x + 99.93

def high_model_b_fit(x):
    return 0.0000060596*x + 99.66998

# --- 6. Generate the Two Final Graphs ---
normal_routes_data = [r for r in results if r['category'] == 'normal']
high_routes_data = [r for r in results if r['category'] == 'high']

create_plot(normal_routes_data, 'Model Accuracy on Normal Latitude Routes (φ_max < 60°)', normal_model_a_fit, normal_model_b_fit)
create_plot(high_routes_data, 'Model Accuracy on High Latitude Routes (φ_max > 60°)', high_model_a_fit, high_model_b_fit)

plt.show()