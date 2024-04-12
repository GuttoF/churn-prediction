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

    def shape(self, table: str) -> "QueryBuilder":
        """
        Retrieves the number of rows and columns of a table.

        Args:
            table (str): The table name.

        Returns:
            self: The QueryBuilder instance.
        """
        self.query = f"SELECT (SELECT COUNT(*) FROM {table}) AS row_count, (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}') AS column_count"
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
        # We need to ensure the SQL is correct. Using `PRAGMA` to get column info in DuckDB
        self.query = f"PRAGMA table_info({table})"
        return self

    def get_types(self, conn, table: str) -> list:
        """
        Executes the SQL query to get column info and extracts only the column types.

        Args:
        - conn: The database connection object.
        - table: A string representing the table name.

        Returns:
        - A list containing only the types of the columns.
        """
        # Build and execute the query to retrieve all column information
        query = self.column_types(table).build()
        result = conn.execute(query).fetchall()
        # Extract only the column types from the result
        return [
            [(col[1], col[2]) for col in result]
        ]  # Index 2 is where 'type' is located based on PRAGMA output

    def count_nulls(self, table: str) -> "QueryBuilder":
        """
        Constructs a query to count null values across all columns in the specified table.

        Args:
            table (str): The table name.

        Returns:
            self: The QueryBuilder instance.
        """
        self.query = f"SELECT "
        # This subquery retrieves all column names from the specified table
        # and constructs a series of COUNT(*) WHERE column IS NULL for each.
        subquery = f"(SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}')"
        # Use the subquery to dynamically generate the outer query part
        self.query += f"(SELECT STRING_AGG('COUNT(CASE WHEN ' || COLUMN_NAME || ' IS NULL THEN 1 END) AS ' || COLUMN_NAME, ', ') FROM {subquery}) AS query_part;"
        return self

    def build_and_execute_count_nulls(self, conn, table: str):
        """
        Builds and executes the query to count null values for all columns in a table.

        Args:
            conn: The database connection object.
            table (str): The table name.

        Returns:
            A result set with the count of nulls for each column.
        """
        # Build the query to get the SQL for counting nulls
        null_count_query = self.count_nulls(table).build()
        # Execute the query to get the actual SQL statement from the aggregation
        sql_for_null_counts = conn.execute(null_count_query).fetchone()[0]
        # Now execute the SQL statement to count nulls across all columns
        final_result = conn.execute(f"SELECT {sql_for_null_counts} FROM {table};").df()
        return final_result

    def count_zeros(self, table: str) -> "QueryBuilder":
        """
        Constructs a query to count zero values across all columns of specific types (BIGINT or DOUBLE) in the specified table.

        Args:
            table (str): The table name.

        Returns:
            self: The QueryBuilder instance.
        """
        self.query = "SELECT "
        # This subquery retrieves column names of type BIGINT or DOUBLE from the specified table
        subquery = f"(SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' AND DATA_TYPE IN ('BIGINT', 'DOUBLE'))"
        # Use the subquery to dynamically generate the outer query part
        self.query += f"(SELECT STRING_AGG('COUNT(CASE WHEN ' || COLUMN_NAME || ' = 0 THEN 1 END) AS ' || COLUMN_NAME, ', ') FROM {subquery}) AS query_part;"
        return self

    def build_and_execute_count_zeros(self, conn, table: str):
        """
        Builds and executes the query to count zero values for all columns of specific types (BIGINT or DOUBLE) in a table.

        Args:
            conn: The database connection object.
            table (str): The table name.

        Returns:
            A result set with the count of zeros for each column.
        """
        # Build the query to get the SQL for counting zeros
        zero_count_query = self.count_zeros(table).build()
        # Execute the query to get the actual SQL statement from the aggregation
        sql_for_zero_counts = conn.execute(zero_count_query).fetchone()[0]
        # Now execute the SQL statement to count zeros across all columns of specific types
        final_result = conn.execute(f"SELECT {sql_for_zero_counts} FROM {table};").df()
        return final_result

    def build(self) -> "QueryBuilder":
        """
        Builds the final query string and resets the query.

        Returns:
        - final_query: The final query string.
        """
        final_query = self.query.rstrip() + ";"
        self.reset()
        return final_query
