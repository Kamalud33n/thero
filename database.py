"""
database.py
Centralized database configuration for the Rehabilitation AI System.

- Loads DATABASE_URL (and any other secrets) from a local .env file.
- If pointed at MySQL, auto-creates the target database on startup so a
  fresh MySQL install doesn't need any manual `CREATE DATABASE` step —
  just create the user/grant privileges once, then run the app.
- Also auto-syncs column TYPES on every startup (e.g. if a model column
  is widened in code, like BLOB -> LONGBLOB), so teammates can `git pull`
  and run the app without hand-running any ALTER TABLE / migration step.
- Exposes: engine, SessionLocal, Base, get_db(), init_db()
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

# ─── Load .env ──────────────────────────────────────────────────────────────
# Looks for a .env file in the current working directory (same folder as
# app.py). Safe to call even if the file doesn't exist — falls back to
# whatever real environment variables / defaults are already set.
load_dotenv()

# ─── Database URL ───────────────────────────────────────────────────────────
# .env example:
#   DATABASE_URL=mysql+pymysql://root:2002@localhost:3306/rehab_db?charset=utf8mb4
# Falls back to local SQLite if DATABASE_URL isn't set at all.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/rehab.db")

# 🔍 Debug: print exactly what URL got loaded (password masked) so it's
# obvious if .env isn't being picked up, or a stale OS-level env var is
# overriding it.
_debug_url = make_url(DATABASE_URL)
print(f"🔍 DATABASE_URL resolved to: "
      f"{_debug_url.drivername}://{_debug_url.username}:***@"
      f"{_debug_url.host}:{_debug_url.port}/{_debug_url.database}")


def _ensure_mysql_database_exists(db_url: str) -> None:
    """
    If DATABASE_URL points at MySQL, connect to the server WITHOUT selecting
    a database and run `CREATE DATABASE IF NOT EXISTS <name>`. This means
    the very first time the app runs against a fresh MySQL server, it
    creates its own database automatically — no manual SQL needed.

    Uses pymysql directly (not a SQLAlchemy engine/URL) for this one-off
    connection, since pymysql.connect() simply omits the `database` kwarg
    when there isn't one — no ambiguity about whether a db got selected.
    """
    import pymysql

    url = make_url(db_url)
    db_name = url.database
    if not db_name:
        return

    host = url.host or "localhost"
    port = url.port or 3306
    user = url.username
    print(f"🔧 Connecting to MySQL at {host}:{port} as '{user}' "
          f"to ensure database '{db_name}' exists...")

    conn = None
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=url.password or "",
            # no `database=` kwarg at all — connects to the server only
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
        print(f"✅ MySQL database '{db_name}' ready (created if it didn't exist)")
    except Exception as exc:
        print(f"⚠️  Could not auto-create MySQL database '{db_name}': {exc}")
        print("    Check DATABASE_URL host/user/password in your .env file, "
              "and that the MySQL server is running.")
        raise
    finally:
        if conn is not None:
            conn.close()


# ─── Engine ──────────────────────────────────────────────────────────────────
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_size=5,
        max_overflow=10,
    )
else:
    _ensure_mysql_database_exists(DATABASE_URL)
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,   # avoid MySQL "server has gone away" on idle conns
        pool_pre_ping=True,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _auto_sync_mysql_columns() -> None:
    """
    Best-effort, safe auto-migration for MySQL.

    After create_all() has built any brand-new tables, walk every existing
    table's columns and compare the TYPE declared in the model (app.py)
    against what's actually sitting in the database. If a model column was
    widened in code (e.g. photo: BLOB -> LONGBLOB), issue an ALTER TABLE to
    bring the live schema in line automatically.

    Deliberately conservative:
    - Only ever widens/changes a column's TYPE — never drops columns,
      renames columns, or drops tables.
    - Skips primary key columns entirely (never touches PK definitions).
    - Preserves the column's existing nullability when altering.
    - Only runs against tables that already exist (new tables are already
      correct straight out of create_all()).

    This means a teammate can clone the repo, set up `.env`, and just run
    the app — no manual SQL / migration step required.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # brand new table — create_all() already built it correctly

            db_columns = {c["name"]: c for c in inspector.get_columns(table.name)}

            for column in table.columns:
                if column.primary_key:
                    continue  # never touch PK columns

                db_col = db_columns.get(column.name)
                if db_col is None:
                    continue  # adding brand-new columns to an existing table is out of scope here

                try:
                    model_type_str = str(column.type.compile(dialect=engine.dialect))
                    db_type_str = str(db_col["type"].compile(dialect=engine.dialect))
                except Exception:
                    continue  # if either type can't compile for this dialect, skip safely

                if model_type_str.upper() == db_type_str.upper():
                    continue  # already in sync

                nullability = "NULL" if column.nullable else "NOT NULL"
                print(f"🔧 Column '{table.name}.{column.name}' out of sync: "
                      f"{db_type_str} -> {model_type_str}. Auto-altering...")
                try:
                    conn.execute(text(
                        f"ALTER TABLE `{table.name}` "
                        f"MODIFY COLUMN `{column.name}` {model_type_str} {nullability}"
                    ))
                    print(f"✅ Column '{table.name}.{column.name}' updated to {model_type_str}")
                except Exception as exc:
                    print(f"⚠️  Could not auto-alter '{table.name}.{column.name}': {exc}")
                    print("    You may need to update it manually with an ALTER TABLE statement.")


def init_db() -> None:
    """
    Create all tables that don't exist yet, then auto-sync column types on
    any tables that already existed (MySQL only). Must be called AFTER all
    model classes (Patient, SessionModel, JointAngle, ...) have been
    imported/defined somewhere, so they're registered on Base.metadata —
    otherwise SQLAlchemy has nothing to create.
    """
    Base.metadata.create_all(bind=engine)
    if not DATABASE_URL.startswith("sqlite"):
        _auto_sync_mysql_columns()
    print(f"✅ Database tables ready ({'MySQL' if not DATABASE_URL.startswith('sqlite') else 'SQLite'})")


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()