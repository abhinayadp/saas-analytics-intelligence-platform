import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["saas_analytics"]
collection = db["accounts_master"]

if __name__ == "__main__":

    # Load master dataset with risk scores
    master = pd.read_csv("master_with_risk.csv")

    records = master.to_dict(orient="records")

    collection.delete_many({})
    collection.insert_many(records)

    print("Updated MongoDB with risk scores.")