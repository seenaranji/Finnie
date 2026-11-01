"""Step 15b lessons you can see: float vs NUMERIC, SQL injection (unsafe vs safe), and a transaction rolling back.

Everything happens on a TEMP table inside one connection, so nothing real can be changed or dropped.

Run:  .venv/bin/python -m scripts.step15b_sql_demo
"""
import psycopg

from src.core.config import Secrets

with psycopg.connect(Secrets().require_database_url(), autocommit=True) as connection:
    print("1. money in floats vs NUMERIC")
    print("   Python float: 0.1 + 0.2 =", 0.1 + 0.2)
    print("   PostgreSQL:   0.1::float8 + 0.2::float8 =", connection.execute("SELECT 0.1::float8 + 0.2::float8").fetchone()[0],
          "   but NUMERIC 0.1 + 0.2 =", connection.execute("SELECT 0.1::numeric + 0.2::numeric").fetchone()[0])

    connection.execute("CREATE TEMP TABLE demo_portfolios (id serial PRIMARY KEY, name text)")
    connection.execute("INSERT INTO demo_portfolios (name) VALUES ('alice retirement'), ('bob growth'), ('carol income')")

    print("\n2. SQL injection")
    attack = "x' OR '1'='1"
    unsafe = f"SELECT name FROM demo_portfolios WHERE name = '{attack}'"          # user text pasted INTO the SQL
    print("   UNSAFE f-string builds:", unsafe)
    print("   returns:", [r[0] for r in connection.execute(unsafe)], " <- every row, though no portfolio is called that")
    safe_rows = connection.execute("SELECT name FROM demo_portfolios WHERE name = %s", (attack,)).fetchall()
    print("   SAFE placeholder, same text passed as a value -> returns:", safe_rows, "(it is just a name nobody has)")
    drop = "x'; DROP TABLE demo_portfolios; --"
    connection.execute(f"SELECT name FROM demo_portfolios WHERE name = '{drop}'")   # unsafe on purpose: the second statement runs
    still_there = connection.execute("SELECT to_regclass('pg_temp.demo_portfolios')").fetchone()[0]
    print(f"   UNSAFE with '{drop}' -> table exists afterwards? {still_there is not None}   <- the table was dropped")

    print("\n3. a transaction rolls back")
    connection.execute("CREATE TEMP TABLE demo_holdings (ticker text, shares numeric CHECK (shares > 0))")
    try:
        with connection.transaction():
            connection.execute("INSERT INTO demo_holdings VALUES ('VTI', 10)")
            connection.execute("INSERT INTO demo_holdings VALUES ('BAD', -1)")   # violates the CHECK
    except psycopg.errors.CheckViolation as error:
        print("   second insert failed:", str(error).splitlines()[0])
    print("   rows left after the failed save:", connection.execute("SELECT count(*) FROM demo_holdings").fetchone()[0], "(the first insert was undone too)")
print("\n*** DEMO FINISHED ***")
