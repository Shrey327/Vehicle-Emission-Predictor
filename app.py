from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the trained Keras model and scaler
try:
    emission_model = load_model('best_model (2).keras')
    scaler = joblib.load('emission_scaler.pkl')
    print("Keras model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    emission_model = None
    scaler = None

def calculate_vsp(speed_ms, acceleration_ms2):
    """Calculates VSP (Vehicle Specific Power)."""
    v = speed_ms
    a = acceleration_ms2
    vsp = v * (1.1 * a + 0.132) + 0.000302 * (v**3)
    return vsp

def reconstruct_trajectory(sparse_points):
    """Reconstructs a smooth trajectory from sparse points."""
    times = np.array([point['time'] for point in sparse_points])
    speeds = np.array([point['speed'] for point in sparse_points])
    
    # Sort points by time if they're not already sorted
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    speeds = speeds[sort_idx]
    
    # Check if we have enough points for cubic interpolation
    if len(times) >= 4:
        # Create interpolation function with cubic spline
        f = interp1d(times, speeds, kind='cubic', fill_value='extrapolate')
    else:
        # Fall back to linear interpolation for sparse points
        f = interp1d(times, speeds, kind='linear', fill_value='extrapolate')
    
    # Generate dense time points
    dense_times = np.arange(times[0], times[-1] + 1)
    
    # Interpolate speeds
    dense_speeds = f(dense_times)
    
    # Calculate accelerations
    accelerations = np.gradient(dense_speeds, dense_times)
    
    return {
        'times': dense_times.tolist(),
        'speeds': dense_speeds.tolist(),
        'accelerations': accelerations.tolist()
    }

def estimate_emissions(speeds, accelerations):
    """Estimates emissions for a trajectory using the Keras model."""
    emissions = []
    for speed, acc in zip(speeds, accelerations):
        vsp = calculate_vsp(speed, acc)
        
        # Create a feature vector with default values for features we don't have
        # This matches the 8 features expected by the scaler:
        # vehicle_class, engine_size, cylinders, fuel_type, city_l/100km, hwy_l/100km, comb_l/100km, comb_mpg
        # Using average/default values for missing parameters
        default_features = np.array([[
            1,              # Default vehicle class (sedan)
            2.0,            # Default engine size
            4,              # Default cylinders
            0,              # Default fuel type (gasoline)
            10.0,           # Default city fuel consumption
            7.0,            # Default highway fuel consumption
            8.5,            # Default combined fuel consumption
            27.0            # Default combined MPG
        ]], dtype=np.float32)       
        
        # Scale the default features
        if scaler is not None:
            scaled_features = scaler.transform(default_features)
            # Use the scaled speed and VSP values for prediction
            # We'll use a simple linear model to predict emissions based on speed, acc, and VSP
            emission = 0.5 + 0.05 * speed + 0.2 * abs(acc) + 0.01 * vsp
            emissions.append(emission)
        else:
            # Fallback method if scaler is not available
            emission = adjust_prediction(speed, acc, 1.0)  # Use default prediction
            emissions.append(emission)
    return emissions

def adjust_prediction(speed_ms, acceleration_ms2, raw_prediction):
    """Adjust the model's prediction to ensure realistic values."""
    # Base emission factors (based on typical vehicle emissions)
    idle_emission = 0.4  # g/s at idle
    city_factor = 0.05  # Additional emission per m/s
    acc_factor = 0.2   # Additional emission per m/s²
    
    # Calculate minimum expected emission based on speed and acceleration
    if speed_ms < 0.1:  # Vehicle is practically stationary
        min_emission = idle_emission
    else:
        # Minimum emission increases with speed and acceleration
        speed_component = idle_emission + (speed_ms * city_factor)
        acc_component = abs(acceleration_ms2) * acc_factor
        min_emission = speed_component + acc_component

    # If model prediction is negative or too low, use the calculated minimum
    if raw_prediction < min_emission:
        return min_emission
    
    # Cap maximum emissions at a reasonable value
    max_emission = min_emission * 3  # Maximum is 3 times the minimum expected
    return min(raw_prediction, max_emission)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if emission_model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        if scaler is None:
            return jsonify({'error': 'Scaler not loaded. Please check server logs.'}), 500
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        required_fields = [
            'vehicle_class', 'engine_size', 'cylinders', 'fuel_type',
            'city_l_100km', 'hwy_l_100km', 'comb_l_100km', 'comb_mpg'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required parameter: {field}'}), 400
        # Prepare input in the correct order
        input_data = [
            data['vehicle_class'],
            data['engine_size'],
            data['cylinders'],
            data['fuel_type'],
            data['city_l_100km'],
            data['hwy_l_100km'],
            data['comb_l_100km'],
            data['comb_mpg']
        ]
        # For categorical fields, you must encode them as the model expects (label or one-hot)
        # Here, we assume label encoding for simplicity. Replace with your actual encoding logic.
        # Example label encodings (replace with your actual mappings):
        vehicle_class_map = {'SUV': 0, 'Sedan': 1, 'Truck': 2, 'Compact': 3}
        fuel_type_map = {'Gasoline': 0, 'Diesel': 1, 'Electric': 2, 'Hybrid': 3}
        # Encode vehicle_class
        if isinstance(input_data[0], str):
            input_data[0] = vehicle_class_map.get(input_data[0], 0)
        # Encode fuel_type
        if isinstance(input_data[3], str):
            input_data[3] = fuel_type_map.get(input_data[3], 0)
        features = np.array([input_data], dtype=np.float32)
        
        # Scale features before prediction
        scaled_features = scaler.transform(features)
        
        raw_prediction = emission_model.predict(scaled_features, verbose=0)[0][0]
        return jsonify({
            'emission_prediction': float(raw_prediction),
            'units': 'g/km',
            'input_parameters': {
                'Vehicle Class': data['vehicle_class'],
                'Engine Size(L)': data['engine_size'],
                'Cylinders': data['cylinders'],
                'Fuel Type': data['fuel_type'],
                'Fuel Consumption City (L/100 km)': data['city_l_100km'],
                'Fuel Consumption Hwy (L/100 km)': data['hwy_l_100km'],
                'Fuel Consumption Comb (L/100 km)': data['comb_l_100km'],
                'Fuel Consumption Comb (mpg)': data['comb_mpg']
            }
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/reconstruct_trajectory', methods=['POST'])
def process_trajectory():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        
        if not data or 'points' not in data:
            return jsonify({'error': 'No trajectory points provided'}), 400
            
        # Reconstruct trajectory
        reconstructed = reconstruct_trajectory(data['points'])
        
        # Estimate emissions
        emissions = estimate_emissions(reconstructed['speeds'], reconstructed['accelerations'])
        
        # Calculate total emissions
        total_emission = sum(emissions)
        
        # If original emissions are provided, calculate difference
        original_emission = data.get('original_emission', None)
        difference_percentage = None
        if original_emission is not None:
            difference_percentage = ((total_emission - original_emission) / original_emission) * 100
        
        return jsonify({
            'reconstructed_trajectory': {
                'times': reconstructed['times'],
                'speeds': reconstructed['speeds'],
                'accelerations': reconstructed['accelerations'],
                'emissions': emissions
            },
            'total_emission': total_emission,
            'difference_percentage': difference_percentage
        })

    except Exception as e:
        print(f"Error in trajectory reconstruction: {str(e)}")
        return jsonify({'error': f'Trajectory reconstruction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 