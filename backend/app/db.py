from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://postgres:arjun@localhost:5432/ai_docs"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # 🔹 Import models so tables get registered
    from . import models  
    print("📌 Creating tables if they don’t exist...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables ready!")
