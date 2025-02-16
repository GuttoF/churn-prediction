import os
import re
import sys

import duckdb
from dotenv import load_dotenv


def snake_case(column_name: str) -> str:
    """
    Converts a CamelCase or PascalCase string to snake_case.

    Args:
    - column_name: A string representing the column name.

    Returns:
    - str: The column name in snake_case.
    """
    # Primeiro, inserimos um sublinhado antes das letras maiúsculas, exceto no início
    name_with_underscores = re.sub(r"(?<!^)(?=[A-Z])", "_", column_name)
    # Depois, convertemos tudo para minúsculas
    return name_with_underscores.lower()


# Load the environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=env_path)

path = os.getenv("HOMEPATH")
full_path = os.path.join(path, "data/raw/churn.csv")
new_path = os.path.join(path, "data/interim/churn.db")

# verify if the file exists
if not os.path.exists(full_path):
    print(f"File {full_path} not found.")
    sys.exit()

if os.path.exists(new_path):
    print(f"The file located in {new_path} already exists.")
    sys.exit()

try:
    conn = duckdb.connect(new_path)
    conn.execute(f"CREATE TABLE churn AS SELECT * FROM read_csv_auto('{full_path}')")

    # rename columns to snake case
    columns_info = conn.execute("PRAGMA table_info('churn')").fetchall()
    for column in columns_info:
        old_name = column[1]
        new_name = snake_case(old_name)
        conn.execute(f"ALTER TABLE churn RENAME COLUMN {old_name} TO {new_name}")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit()
finally:
    conn.close()
