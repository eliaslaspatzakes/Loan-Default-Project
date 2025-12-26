import pandas as pd
from sqlalchemy import create_engine,text
import os


connection_str = "postgresql+psycopg2://postgres:1131995i%40@localhost:5432/Bank"
try:
    engine = create_engine(connection_str)
    with engine.connect() as conn:
        print("Successfully connected to PostgreSQL!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()
    
current_script_folder = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(current_script_folder,"bank data")

print(f"Looking for data in: {data_dir}")

if not os.path.exists(data_dir):
    print("ERROR: Still cannot find the data folder.")
    print(f"Please make sure the folder 'data' is inside: {current_script_folder}")
    exit()
    
    
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# --- DATA LOADING ---
for file in files:
    try:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path,sep = ";",low_memory= False)
        df.to_sql(file.replace('.csv', ''), engine, if_exists='replace', index=False)
        print(f"   Success! Added {len(df)} rows.")
    except Exception as e:
        print(f"   Error processing {file}: {e}")
        
print("\nDone! Your PostgreSQL database is ready.")