import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import ExperimentDatabase

MATERIAL_DATA = [
    {
        'category': 'Carbon Fiber Composite',
        'material_name': 'T300 Carbon Fiber',
        'emissivity_mean': 0.88,
        'emissivity_std': 0.03,
        'reflectivity_mean': 0.10,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 10.0,
        'specific_heat': 0.71,
        'density': 1.80,
        'roughness': 0.8,
        'description': 'Standard modulus carbon fiber composite'
    },
    {
        'category': 'Carbon Fiber Composite',
        'material_name': 'T700 Carbon Fiber',
        'emissivity_mean': 0.90,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.08,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 15.0,
        'specific_heat': 0.71,
        'density': 1.80,
        'roughness': 0.6,
        'description': 'High strength carbon fiber composite'
    },
    {
        'category': 'Carbon Fiber Composite',
        'material_name': 'M40 Carbon Fiber',
        'emissivity_mean': 0.85,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.12,
        'reflectivity_std': 0.03,
        'thermal_conductivity': 300.0,
        'specific_heat': 0.71,
        'density': 1.81,
        'roughness': 0.4,
        'description': 'High modulus carbon fiber composite'
    },
    {
        'category': 'High-Hardness Steel',
        'material_name': '440C Stainless Steel',
        'emissivity_mean': 0.25,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.72,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 15.1,
        'specific_heat': 0.46,
        'density': 7.85,
        'roughness': 0.2,
        'description': 'Martensitic stainless steel, high hardness'
    },
    {
        'category': 'High-Hardness Steel',
        'material_name': 'D2 Tool Steel',
        'emissivity_mean': 0.30,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.67,
        'reflectivity_std': 0.04,
        'thermal_conductivity': 11.0,
        'specific_heat': 0.47,
        'density': 7.70,
        'roughness': 0.3,
        'description': 'Cold work tool steel'
    },
    {
        'category': 'High-Hardness Steel',
        'material_name': 'H13 Tool Steel',
        'emissivity_mean': 0.35,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.62,
        'reflectivity_std': 0.04,
        'thermal_conductivity': 25.0,
        'specific_heat': 0.48,
        'density': 7.85,
        'roughness': 0.3,
        'description': 'Hot work tool steel'
    },
    {
        'category': 'High-Hardness Steel',
        'material_name': 'S7 Tool Steel',
        'emissivity_mean': 0.40,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.57,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 18.0,
        'specific_heat': 0.46,
        'density': 7.80,
        'roughness': 0.4,
        'description': 'Shock-resistant tool steel'
    },
    {
        'category': 'Carburized Aluminum',
        'material_name': 'Carburized Al 6061',
        'emissivity_mean': 0.15,
        'emissivity_std': 0.03,
        'reflectivity_mean': 0.82,
        'reflectivity_std': 0.03,
        'thermal_conductivity': 167.0,
        'specific_heat': 0.89,
        'density': 2.70,
        'roughness': 0.2,
        'description': 'Carburized aluminum alloy 6061'
    },
    {
        'category': 'Carburized Aluminum',
        'material_name': 'Carburized Al 7075',
        'emissivity_mean': 0.18,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.79,
        'reflectivity_std': 0.04,
        'thermal_conductivity': 130.0,
        'specific_heat': 0.89,
        'density': 2.81,
        'roughness': 0.2,
        'description': 'Carburized aluminum alloy 7075'
    },
    {
        'category': 'Aluminum Alloy',
        'material_name': 'Al 2024-T3',
        'emissivity_mean': 0.08,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.89,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 190.0,
        'specific_heat': 0.89,
        'density': 2.78,
        'roughness': 0.1,
        'description': 'High strength aluminum alloy'
    },
    {
        'category': 'Aluminum Alloy',
        'material_name': 'Al 3003-H14',
        'emissivity_mean': 0.10,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.87,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 200.0,
        'specific_heat': 0.89,
        'density': 2.73,
        'roughness': 0.15,
        'description': 'Aluminum manganese alloy'
    },
    {
        'category': 'Aluminum Alloy',
        'material_name': 'Al 5052-H32',
        'emissivity_mean': 0.12,
        'emissivity_std': 0.03,
        'reflectivity_mean': 0.85,
        'reflectivity_std': 0.03,
        'thermal_conductivity': 138.0,
        'specific_heat': 0.89,
        'density': 2.68,
        'roughness': 0.2,
        'description': 'Aluminum magnesium alloy'
    },
    {
        'category': 'Aluminum Alloy',
        'material_name': 'Al 6061-T6',
        'emissivity_mean': 0.10,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.87,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 167.0,
        'specific_heat': 0.89,
        'density': 2.70,
        'roughness': 0.15,
        'description': 'Heat treatable aluminum alloy'
    },
    {
        'category': 'Aluminum Alloy',
        'material_name': 'Al 7075-T6',
        'emissivity_mean': 0.08,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.89,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 130.0,
        'specific_heat': 0.89,
        'density': 2.81,
        'roughness': 0.1,
        'description': 'Ultra-high strength aluminum alloy'
    },
    {
        'category': 'Ni-Mo-W Alloy',
        'material_name': 'Hastelloy X',
        'emissivity_mean': 0.35,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.62,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 11.0,
        'specific_heat': 0.44,
        'density': 8.20,
        'roughness': 0.3,
        'description': 'Nickel-chromium-molybdenum alloy'
    },
    {
        'category': 'Ni-Mo-W Alloy',
        'material_name': 'Inconel 718',
        'emissivity_mean': 0.40,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.57,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 11.5,
        'specific_heat': 0.43,
        'density': 8.24,
        'roughness': 0.3,
        'description': 'Nickel-iron-chromium superalloy'
    },
    {
        'category': 'Corroded Steel',
        'material_name': 'Mild Steel (0% Corrosion)',
        'emissivity_mean': 0.35,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.62,
        'reflectivity_std': 0.04,
        'thermal_conductivity': 45.0,
        'specific_heat': 0.45,
        'density': 7.86,
        'roughness': 0.2,
        'description': 'Mild steel with no corrosion'
    },
    {
        'category': 'Corroded Steel',
        'material_name': 'Mild Steel (25% Corrosion)',
        'emissivity_mean': 0.50,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.47,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 40.0,
        'specific_heat': 0.45,
        'density': 7.80,
        'roughness': 0.5,
        'description': 'Mild steel with 25% surface corrosion'
    },
    {
        'category': 'Corroded Steel',
        'material_name': 'Mild Steel (50% Corrosion)',
        'emissivity_mean': 0.65,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.32,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 35.0,
        'specific_heat': 0.45,
        'density': 7.75,
        'roughness': 0.7,
        'description': 'Mild steel with 50% surface corrosion'
    },
    {
        'category': 'Corroded Steel',
        'material_name': 'Mild Steel (75% Corrosion)',
        'emissivity_mean': 0.80,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.17,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 30.0,
        'specific_heat': 0.45,
        'density': 7.70,
        'roughness': 0.9,
        'description': 'Mild steel with 75% surface corrosion'
    },
    {
        'category': 'Anti-Optical Coating',
        'material_name': 'SiO2 Coating',
        'emissivity_mean': 0.08,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.89,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 1.4,
        'specific_heat': 0.70,
        'density': 2.20,
        'roughness': 0.05,
        'description': 'Silicon dioxide anti-reflective coating'
    },
    {
        'category': 'Anti-Optical Coating',
        'material_name': 'MgF2 Coating',
        'emissivity_mean': 0.06,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.91,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 0.4,
        'specific_heat': 1.00,
        'density': 3.18,
        'roughness': 0.05,
        'description': 'Magnesium fluoride anti-reflective coating'
    },
    {
        'category': 'Anti-Optical Coating',
        'material_name': 'TiO2 Coating',
        'emissivity_mean': 0.12,
        'emissivity_std': 0.03,
        'reflectivity_mean': 0.85,
        'reflectivity_std': 0.03,
        'thermal_conductivity': 10.0,
        'specific_heat': 0.71,
        'density': 4.23,
        'roughness': 0.08,
        'description': 'Titanium dioxide coating'
    },
    {
        'category': 'Anti-Infrared Coating',
        'material_name': 'IR-Absorbing Paint',
        'emissivity_mean': 0.92,
        'emissivity_std': 0.03,
        'reflectivity_mean': 0.05,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 0.15,
        'specific_heat': 1.20,
        'density': 1.20,
        'roughness': 0.5,
        'description': 'Infrared absorbing coating'
    },
    {
        'category': 'Anti-Infrared Coating',
        'material_name': 'Pyrolytic Carbon',
        'emissivity_mean': 0.95,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.03,
        'reflectivity_std': 0.01,
        'thermal_conductivity': 200.0,
        'specific_heat': 0.71,
        'density': 2.00,
        'roughness': 0.3,
        'description': 'Pyrolytic carbon coating'
    },
    {
        'category': 'Anti-Infrared Coating',
        'material_name': 'Black Chrome',
        'emissivity_mean': 0.88,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.09,
        'reflectivity_std': 0.03,
        'thermal_conductivity': 60.0,
        'specific_heat': 0.45,
        'density': 7.80,
        'roughness': 0.2,
        'description': 'Black chrome coating'
    },
    {
        'category': 'Polyurethane Coating',
        'material_name': 'PU Clear Coat',
        'emissivity_mean': 0.88,
        'emissivity_std': 0.03,
        'reflectivity_mean': 0.09,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 0.20,
        'specific_heat': 1.80,
        'density': 1.10,
        'roughness': 0.1,
        'description': 'Clear polyurethane coating'
    },
    {
        'category': 'Polyurethane Coating',
        'material_name': 'PU Black Coat',
        'emissivity_mean': 0.92,
        'emissivity_std': 0.02,
        'reflectivity_mean': 0.05,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 0.18,
        'specific_heat': 1.80,
        'density': 1.15,
        'roughness': 0.2,
        'description': 'Black polyurethane coating'
    },
    {
        'category': 'Polyimide Film',
        'material_name': 'Kapton HN',
        'emissivity_mean': 0.48,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.49,
        'reflectivity_std': 0.04,
        'thermal_conductivity': 0.12,
        'specific_heat': 1.09,
        'density': 1.42,
        'roughness': 0.15,
        'description': 'Kapton polyimide film'
    },
    {
        'category': 'Polyimide Film',
        'material_name': 'Upilex S',
        'emissivity_mean': 0.52,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.45,
        'reflectivity_std': 0.04,
        'thermal_conductivity': 0.15,
        'specific_heat': 1.05,
        'density': 1.43,
        'roughness': 0.12,
        'description': 'High temperature polyimide film'
    },
    {
        'category': 'Ceramic Coating',
        'material_name': 'Al2O3 Ceramic',
        'emissivity_mean': 0.88,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.09,
        'reflectivity_std': 0.03,
        'thermal_conductivity': 25.0,
        'specific_heat': 0.77,
        'density': 3.95,
        'roughness': 0.3,
        'description': 'Alumina ceramic coating'
    },
    {
        'category': 'Ceramic Coating',
        'material_name': 'ZrO2 Ceramic',
        'emissivity_mean': 0.92,
        'emissivity_std': 0.03,
        'reflectivity_mean': 0.05,
        'reflectivity_std': 0.02,
        'thermal_conductivity': 2.0,
        'specific_heat': 0.60,
        'density': 5.89,
        'roughness': 0.4,
        'description': 'Zirconia ceramic coating'
    },
    {
        'category': 'Titanium Alloy',
        'material_name': 'Ti-6Al-4V',
        'emissivity_mean': 0.40,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.57,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 7.0,
        'specific_heat': 0.52,
        'density': 4.43,
        'roughness': 0.25,
        'description': 'Alpha-beta titanium alloy'
    },
    {
        'category': 'Titanium Alloy',
        'material_name': 'Ti-5Al-5Mo-5V-3Cr',
        'emissivity_mean': 0.45,
        'emissivity_std': 0.05,
        'reflectivity_mean': 0.52,
        'reflectivity_std': 0.05,
        'thermal_conductivity': 6.5,
        'specific_heat': 0.50,
        'density': 4.80,
        'roughness': 0.3,
        'description': 'Beta titanium alloy'
    },
    {
        'category': 'Titanium Alloy',
        'material_name': 'Ti-3Al-2.5V',
        'emissivity_mean': 0.38,
        'emissivity_std': 0.04,
        'reflectivity_mean': 0.59,
        'reflectivity_std': 0.04,
        'thermal_conductivity': 10.0,
        'specific_heat': 0.52,
        'density': 4.48,
        'roughness': 0.2,
        'description': 'Alpha titanium alloy'
    }
]

# --------------------------------------------------------------------------- #
# Per-category temperature range (K), consistent with manuscript Table 1
# (tab:material_database). Attached to each material so data generators can
# sample temperature within the material's valid range (论文 B-1: 避免在
# 1273 K 下生成 Carbon Fiber 等越界样本).
# --------------------------------------------------------------------------- #
CATEGORY_TEMP_RANGE = {
    'Carbon Fiber Composite': (273.0, 673.0),
    'High-Hardness Steel':    (273.0, 873.0),
    'Carburized Aluminum':    (273.0, 573.0),
    'Aluminum Alloy':         (273.0, 573.0),
    'Ni-Mo-W Alloy':          (273.0, 973.0),
    'Corroded Steel':         (273.0, 773.0),
    'Anti-Optical Coating':   (273.0, 473.0),
    'Anti-Infrared Coating':  (273.0, 473.0),
    'Polyurethane Coating':   (273.0, 373.0),
    'Polyimide Film':         (273.0, 473.0),
    'Ceramic Coating':        (273.0, 1273.0),
    'Titanium Alloy':         (273.0, 873.0),
}

for _material in MATERIAL_DATA:
    _material.setdefault('temp_range', CATEGORY_TEMP_RANGE[_material['category']])


def initialize_database(db_path: str = "./data/experiment.db"):
    """初始化材料数据库"""
    print(f"Initializing database at: {db_path}")
    
    db = ExperimentDatabase(db_path)
    
    print(f"Adding {len(MATERIAL_DATA)} materials to database...")
    db.add_materials_batch(MATERIAL_DATA)
    
    materials = db.get_materials()
    categories = db.get_all_categories()
    
    print(f"\nDatabase initialized successfully!")
    print(f"Total materials: {len(materials)}")
    print(f"Total categories: {len(categories)}")
    print(f"\nCategories:")
    for cat in categories:
        cat_materials = db.get_materials(cat)
        print(f"  - {cat}: {len(cat_materials)} materials")
    
    return db

if __name__ == "__main__":
    initialize_database()