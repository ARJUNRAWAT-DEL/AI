# backend/app/init_db.py

from .db import engine, Base
from .models import *


print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("Done!")
