import pandas as pd

# Load existing car dataset
try:
  df = pd.read_csv('car_data.csv')
except FileNotFoundError:
  print('Error: car_data.csv file pawa jacche na!')
  df = pd.DataFrame(
      columns=[
          'brand',
          'model',
          'milage',
          'car_age',
          'fuel_type',
          'transmission',
          'accident',
          'price',
      ]
  )

# New Indian cars data
new_cars = pd.DataFrame([
    {
        'brand': 'Tata',
        'model': 'Nexon',
        'milage': 25000,
        'car_age': 2,
        'fuel_type': 'Petrol',
        'transmission': 'Manual',
        'accident': 'None',
        'price': 900000,
    },
    {
        'brand': 'Mahindra',
        'model': 'Thar',
        'milage': 15000,
        'car_age': 1,
        'fuel_type': 'Diesel',
        'transmission': 'Manual',
        'accident': 'None',
        'price': 1350000,
    },
    {
        'brand': 'Maruti Suzuki',
        'model': 'Swift',
        'milage': 40000,
        'car_age': 4,
        'fuel_type': 'Petrol',
        'transmission': 'Manual',
        'accident': 'None',
        'price': 600000,
    },
])

# Append new cars to existing dataframe
df = pd.concat([df, new_cars], ignore_index=True)

# Save back to car_data.csv
df.to_csv('car_data.csv', index=False)
print('SUCCESS: Indian cars added to car_data.csv successfully!')