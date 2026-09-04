"""
实验运行示例脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_system import ExperimentSystem, DEFAULT_CONFIG
from experiment_system.config import ExperimentConfig, ProcessingConfig

def run_standard_experiment():
    """运行标准实验"""
    config = DEFAULT_CONFIG
    
    config.experiment = ExperimentConfig(
        experiment_name="standard_non_cooperative_target_test",
        output_dir="./data/standard_test",
        duration_seconds=300,
        temperature_range=[20.0, 80.0],
        temperature_step=10.0
    )
    
    config.processing = ProcessingConfig(
        apply_temperature_compensation=True,
        apply_filtering=True,
        filter_type="ns_arkf",
        apply_inversion=True,
        save_raw_data=True,
        save_processed_data=True
    )
    
    system = ExperimentSystem(config)
    
    print("Running standard experiment...")
    system.run_full_experiment()
    
    print("Experiment completed successfully!")

def run_temperature_sweep():
    """运行温度扫描实验"""
    temperatures = [20, 30, 40, 50, 60, 70, 80]
    
    for temp in temperatures:
        config = DEFAULT_CONFIG
        
        config.experiment = ExperimentConfig(
            experiment_name=f"temperature_sweep_{temp}c",
            output_dir=f"./data/temperature_sweep/{temp}c",
            duration_seconds=120
        )
        
        config.processing = ProcessingConfig(
            apply_temperature_compensation=True,
            apply_filtering=True,
            filter_type="ns_arkf"
        )
        
        print(f"\n{'='*60}")
        print(f"Running experiment at {temp}°C")
        print(f"{'='*60}")
        
        system = ExperimentSystem(config)
        
        try:
            system.setup_sensors()
            system.setup_filters()
            system.calibrate_temperature_compensation()
            
            if system.connect_sensors():
                system.start_experiment()
                system.disconnect_sensors()
                
                system.process_data()
                system.save_data()
                system.analyze_data()
                system.visualize_data()
                
        except Exception as e:
            print(f"Error at {temp}°C: {e}")
            system.disconnect_sensors()

def run_material_test():
    """运行材料测试实验"""
    materials = [
        "carbon_fiber_composite",
        "high_hardness_steel", 
        "carburized_aluminum",
        "aluminum_alloy",
        "ceramic_coating"
    ]
    
    for material in materials:
        config = DEFAULT_CONFIG
        
        config.experiment = ExperimentConfig(
            experiment_name=f"material_test_{material}",
            output_dir=f"./data/material_test/{material}",
            duration_seconds=180
        )
        
        system = ExperimentSystem(config)
        
        print(f"\n{'='*60}")
        print(f"Testing material: {material}")
        print(f"{'='*60}")
        
        try:
            system.setup_sensors()
            system.setup_filters()
            system.calibrate_temperature_compensation()
            
            if system.connect_sensors():
                print("Adjust target material to: ", material)
                input("Press Enter to continue...")
                
                system.start_experiment()
                system.disconnect_sensors()
                
                system.process_data()
                system.save_data()
                system.analyze_data()
                system.visualize_data()
                
        except Exception as e:
            print(f"Error testing {material}: {e}")
            system.disconnect_sensors()

if __name__ == "__main__":
    print("Non-Cooperative Target Measurement Experiment System")
    print("=" * 60)
    print()
    print("Available experiments:")
    print("1. Standard Experiment")
    print("2. Temperature Sweep (20-80°C)")
    print("3. Material Test")
    print()
    
    choice = input("Select experiment (1/2/3): ").strip()
    
    if choice == "1":
        run_standard_experiment()
    elif choice == "2":
        run_temperature_sweep()
    elif choice == "3":
        run_material_test()
    else:
        print("Invalid choice. Running standard experiment...")
        run_standard_experiment()