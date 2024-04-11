import duckdb


class QueryBuilder:
    """
    A class that helps in building SQL queries in duckdb.

    Methods:
    - reset: Resets the query string.
    - select: Adds the SELECT clause to the query.
    - get_connection: Establishes a connection to the database.
    - from_table: Adds the FROM clause to the query.
    - where: Adds the WHERE clause to the query.
    - group_by: Adds the GROUP BY clause to the query.
    - limit: Adds the LIMIT clause to the query.
    - check_zero: Adds a custom clause to check for zero values in a table.
    - column_names: Adds a custom clause to retrieve column names from a table.
    - column_types: Adds a custom clause to retrieve column names and data types from a table.
    - build: Builds the final query string and resets the query.

    Usage:
    ```
    query_builder = QueryBuilder()
    query = query_builder.select("column1, column2").from_table("table").where("condition").build()
    ```
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the query string.
        """
        self.query = ""

    def get_connection(self, conn_path: str) -> "duckdb.Connection":
        """
        Establishes a connection to the specified database.

        Args:
            conn_path (str): The path to the database file.

        Returns:
            duckdb.Connection: The connection object.

        Raises:
            Exception: If an error occurs while establishing the connection.
        """
        conn = None
        try:
            conn = duckdb.connect(database=conn_path, read_only=False)
        except Exception as e:
            print(e)
        return conn

    def select(self, args: str) -> "QueryBuilder":
        """
        Adds the SELECT clause to the query.

        Args:
        - args: A string representing the columns to select.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query = f"SELECT {args}"
        return self

    def from_table(self, table: str) -> "QueryBuilder":
        """
        Adds the FROM clause to the query.

        Args:
        - table: A string representing the table name.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query += f" FROM {table}"
        return self

    def where(self, condition: str) -> "QueryBuilder":
        """
        Adds the WHERE clause to the query.

        Args:
        - condition: A string representing the condition.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query += f" WHERE {condition}"
        return self

    def group_by(self, args: str) -> "QueryBuilder":
        """
        Adds the GROUP BY clause to the query.

        Args:
        - args: A string representing the columns to group by.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query += f" GROUP BY {args}"
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """
        Adds the LIMIT clause to the query.

        Args:
        - n: An integer representing the limit.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query += f" LIMIT {n}"
        return self

    def check_zero(self, table: str) -> "QueryBuilder":
        """
        Adds a custom clause to check for zero values in a table.

        Args:
        - table: A string representing the table name.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query += f"SUM(CASE WHEN {table} = 0 THEN 1 ELSE 0 END) AS zero_values"
        return self

    def check_null(self, table: str) -> "QueryBuilder":
        """
        Adds a custom clause to check for null values in a table.

        Args:
        - table: A string representing the table name.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query += f"SUM(CASE WHEN {table} IS NULL THEN 1 ELSE 0 END) AS null_values"
        return self

    def column_names(self, table: str) -> "QueryBuilder":
        """
        Adds a custom clause to retrieve column names from a table.

        Args:
        - table: A string representing the table name.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query += f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'"
        return self

    def column_types(self, table: str) -> "QueryBuilder":
        """
        Adds a custom clause to retrieve column names and data types from a table.

        Args:
        - table: A string representing the table name.

        Returns:
        - self: The QueryBuilder instance.
        """
        self.query += f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'"
        return self

    def build(self) -> "QueryBuilder":
        """
        Builds the final query string and resets the query.

        Returns:
        - final_query: The final query string.
        """
        final_query = self.query.rstrip() + ";"
        self.reset()
        return final_query
