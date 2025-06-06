# Vehicle Emission Predictor 🚗💨

A sophisticated web application that predicts vehicle carbon emissions using machine learning. The app can estimate emissions based on vehicle characteristics and reconstruct full trajectories from sparse GPS data to calculate real-world driving emissions.

## 🌟 Features

- **Vehicle Emission Prediction**: Predict CO2 emissions based on vehicle specifications
- **Trajectory Reconstruction**: Convert sparse GPS points into smooth, dense trajectories
- **Real-time Emission Estimation**: Calculate emissions for reconstructed driving patterns
- **Machine Learning Powered**: Uses TensorFlow/Keras neural networks for accurate predictions
- **RESTful API**: Easy integration with other applications
- **Web Interface**: User-friendly web UI for direct interaction

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: TensorFlow/Keras, scikit-learn
- **Data Processing**: NumPy, Pandas, SciPy
- **Model Persistence**: Joblib
- **API**: RESTful endpoints with JSON responses

## 📋 Prerequisites

- Python 3.8 or higher
- Windows 10/11 (for current setup)
- Virtual environment support

## 🚀 Installation & Setup

### 1. Clone/Download the Project
Ensure you have all project files in your directory:
```
Mini Project/
├── app.py
├── requirements.txt
├── best_model (2).keras
├── emission_scaler.pkl
├── templates/
├── static/
└── venv/
```

### 2. Activate Virtual Environment
```bash
# Windows
.\venv\Scripts\Activate

# You should see (venv) in your prompt
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
Test if TensorFlow loads correctly:
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### 5. Run the Application
```bash
python app.py
```

The application will start at: **http://127.0.0.1:5000/**

## 📖 Usage

### Web Interface
1. Open your browser and navigate to `http://127.0.0.1:5000/`
2. Use the web interface to:
   - Input vehicle characteristics for emission prediction
   - Upload trajectory data for reconstruction and analysis

### API Endpoints

#### 🔍 Vehicle Emission Prediction
**Endpoint**: `POST /predict`

Predicts emissions based on vehicle characteristics.

**Request Body**:
```json
{
  "vehicle_class": "Sedan",
  "engine_size": 2.0,
  "cylinders": 4,
  "fuel_type": "Gasoline",
  "city_l_100km": 9.5,
  "hwy_l_100km": 7.2,
  "comb_l_100km": 8.1,
  "comb_mpg": 29.1
}
```

**Response**:
```json
{
  "emission_prediction": 180.5,
  "units": "g/km",
  "input_parameters": {
    "Vehicle Class": "Sedan",
    "Engine Size(L)": 2.0,
    ...
  }
}
```

#### 🛣️ Trajectory Reconstruction
**Endpoint**: `POST /reconstruct_trajectory`

Reconstructs smooth trajectories from sparse GPS points and estimates emissions.

**Request Body**:
```json
{
  "points": [
    {"time": 0, "speed": 0},
    {"time": 30, "speed": 15},
    {"time": 60, "speed": 25},
    {"time": 120, "speed": 10}
  ],
  "original_emission": 1500.0
}
```

**Response**:
```json
{
  "reconstructed_trajectory": {
    "times": [0, 1, 2, ...],
    "speeds": [0, 0.5, 1.0, ...],
    "accelerations": [0, 0.5, 0.5, ...],
    "emissions": [0.4, 0.45, 0.5, ...]
  },
  "total_emission": 1623.45,
  "difference_percentage": 8.23
}
```

## 🧮 Vehicle Specific Power (VSP) Calculation

The application uses the VSP formula for realistic emission estimation:
```
VSP = v × (1.1 × a + 0.132) + 0.000302 × v³
```
Where:
- `v` = vehicle speed (m/s)
- `a` = acceleration (m/s²)

## 📁 Model Files

The application requires these trained model files:
- **`best_model (2).keras`**: TensorFlow/Keras neural network model
- **`emission_scaler.pkl`**: Feature scaler for input normalization

## 🔧 Vehicle Class & Fuel Type Mappings

**Vehicle Classes**:
- 0: SUV
- 1: Sedan  
- 2: Truck
- 3: Compact

**Fuel Types**:
- 0: Gasoline
- 1: Diesel
- 2: Electric
- 3: Hybrid

## 🐛 Troubleshooting

### TensorFlow DLL Issues
If you encounter TensorFlow import errors:
1. Install Microsoft Visual C++ Redistributable
2. Ensure you're using 64-bit Python
3. Try reinstalling TensorFlow: `pip uninstall tensorflow && pip install tensorflow`

### Model Loading Issues
- Verify `best_model (2).keras` and `emission_scaler.pkl` are in the project directory
- Check file permissions and ensure files aren't corrupted

### Common Error Messages
- **"Model not loaded"**: Check if model files exist and are accessible
- **"Scaler not loaded"**: Verify `emission_scaler.pkl` is present
- **"Missing required parameter"**: Ensure all required fields are provided in API requests

## 🚀 Production Deployment

For production deployment, consider:
- Using a WSGI server like Gunicorn or uWSGI
- Setting up a reverse proxy with Nginx
- Implementing proper logging and monitoring
- Adding authentication and rate limiting

**Note**: This is a development server. For production use, please use a proper WSGI server and follow security best practices. 
