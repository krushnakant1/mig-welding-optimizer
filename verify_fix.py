import subprocess
import requests
import time
import sys
import os

def test_app():
    print("Starting welding_app.py...")
    # Adjust path if necessary
    app_path = r"d:\Projects\Machine Learning\FInel Year Project\OPTIMAISATION\welding_app.py"
    process = subprocess.Popen([sys.executable, app_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(5)
    
    base_url = "http://localhost:8000"
    
    try:
        # 1. Generate Demo Data
        print("Testing /api/generate-demo...")
        resp = requests.post(f"{base_url}/api/generate-demo", json={"voltage": 22})
        resp.raise_for_status()
        data = resp.json()
        if not data['success'] or len(data['data']) != 80:
            print("FAILED: Generate demo data")
            return
        training_data = data['data']
        print(f"SUCCESS: Generated {len(training_data)} samples.")

        # 2. Train Model
        print("Testing /api/train-model...")
        resp = requests.post(f"{base_url}/api/train-model", json={"data": training_data})
        resp.raise_for_status()
        res = resp.json()
        if not res['success']:
            print(f"FAILED: Train model - {res.get('error')}")
            return
        print(f"SUCCESS: Model trained. R2 Tensile: {res['cv_scores']['tensile_mean']}, R2 Pen: {res['cv_scores']['pen_mean']}")

        # 3. Predict
        print("Testing /api/predict...")
        payload = {
            "current": 120,
            "voltage": 22,
            "speed": 100,
            "filler": "ER309L",
            "interpass": 25,
            "efficiency": 0.6
        }
        resp = requests.post(f"{base_url}/api/predict", json=payload)
        resp.raise_for_status()
        res = resp.json()
        if not res['success']:
             print(f"FAILED: Predict - {res.get('error')}")
             return
        print(f"SUCCESS: Prediction: {res['prediction']}")

        # 4. Optimize
        print("Testing /api/optimize...")
        opt_payload = {
            "max_heat_input": 1.2,
            "plate_thickness": 3.0,
            "current_min": 80,
            "current_max": 150,
            "speed_min": 80,
            "speed_max": 200,
            "step": 10
        }
        resp = requests.post(f"{base_url}/api/optimize", json=opt_payload)
        resp.raise_for_status()
        res = resp.json()
        if not res['success']:
             print(f"FAILED: Optimize - {res.get('error')}")
             return
        print(f"SUCCESS: Optfound: {res['optimal']}")

        print("\nALL TESTS PASSED!")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        print("Terminating server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    test_app()
