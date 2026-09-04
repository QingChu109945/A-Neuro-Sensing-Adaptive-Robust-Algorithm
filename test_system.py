import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_system import ExperimentSystem, DEFAULT_CONFIG

DEFAULT_CONFIG.experiment.duration_seconds = 2
DEFAULT_CONFIG.experiment.output_dir = "./data/test"

print("Testing Experiment System...")
print("=" * 60)

try:
    system = ExperimentSystem(DEFAULT_CONFIG)
    print("1. ExperimentSystem created successfully")
    
    success = system.initialize_material_database()
    print(f"2. Material database initialized: {success}")
    
    materials = system.storage.get_materials()
    print(f"3. Loaded {len(materials)} materials")
    
    if materials:
        print("\nMaterials list:")
        for m in materials[:5]:
            print(f"  ID:{m['id']:3d} | {m['material_name']:30s} | emissivity: {m['emissivity_mean']:.4f}")
    
    categories = system.storage.get_material_categories()
    print(f"\n4. Material categories: {categories}")
    
    print("\n5. Testing experiment workflow...")
    system.setup_filters()
    print("   Filters setup completed")
    
    system.calibrate_temperature_compensation()
    print("   Temperature compensation calibrated")
    
    system.storage.start_experiment("test_experiment", DEFAULT_CONFIG, 2)
    print("   Experiment started")
    
    system.setup_sensors()
    system.connect_sensors()
    print("   Sensors connected")
    
    system.add_noise_configuration(
        noise_type='gaussian',
        param_1=0.01,
        param_2=42,
        description="Test noise config"
    )
    print("   Noise configuration added")
    
    system.add_measurement_conditions()
    print("   Measurement conditions added")
    
    system.start_experiment()
    print("   Data collection completed")
    
    system.process_data()
    print("   Data processing completed")
    
    system.save_data()
    print("   Data saved")
    
    system.perform_inversion()
    print("   Inversion completed")
    
    system.analyze_data()
    print("   Data analysis completed")
    
    system.storage.end_experiment('completed')
    print("   Experiment completed")
    
    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
