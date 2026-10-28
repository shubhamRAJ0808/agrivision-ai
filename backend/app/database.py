import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("AGRIVISION_DB") or os.path.join(HERE, "..", "agrivision.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
