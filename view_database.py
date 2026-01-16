"""
View Face Recognition Database
==============================
Shows all persons stored in the database.
"""

import pickle
from pathlib import Path

DB_PATH = Path(__file__).parent / 'recognition_database_arcface.pkl'

def view_database():
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
    
    print("=" * 50)
    print("       FACE RECOGNITION DATABASE")
    print("=" * 50)
    print(f"\nTotal persons: {len(db['persons'])}\n")
    print("-" * 50)
    
    for i, (key, data) in enumerate(db['persons'].items(), 1):
        parts = key.split('_', 1)
        member_id = parts[0]
        name = parts[1].replace('_', ' ') if len(parts) > 1 else key
        
        num_embeddings = len(data.get('embeddings', []))
        has_mean = 'mean_embedding' in data
        
        print(f"{i:2}. {name}")
        print(f"    ID: {member_id}")
        print(f"    Embeddings: {num_embeddings}")
        print(f"    Mean embedding: {'Yes' if has_mean else 'No'}")
        print()

if __name__ == "__main__":
    view_database()
