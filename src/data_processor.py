from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col,
    hour,
    sum,
    avg,
    to_timestamp,
    date_trunc,
    round,
    bround,
    explode,
    abs as spark_abs,
    first,
    quarter,
    year,
    concat,
    lit,
    count,
    month,
    date_format,
)

from typing import Dict
import glob
import shutil
import os


class DataProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark

        # Note: all functions are helper functions, you do not have to use them in the main code

    # Load Data from MySQL and MongoDB helper functions
    def fetch_mysql_data(
        self, jdbc_url: str, db_table: str, db_user: str, db_password: str
    ) -> DataFrame:
        """
        Load data from MySQL database.

        :param jdbc_url: JDBC URL for the MySQL database
        :param db_table: Name of the table to load data from
        :param db_user: Database username
        :param db_password: Database password
        :return: DataFrame containing the loaded MySQL data
        """
        return (
            self.spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("driver", "com.mysql.cj.jdbc.Driver")
            .option("dbtable", db_table)
            .option("user", db_user)
            .option("password", db_password)
            .load()
        )

    def fetch_mongo_data(self, db_name: str, collection_name: str) -> DataFrame:
        """
        Load data from MongoDB.

        :param db_name: Name of the MongoDB database
        :param collection_name: Name of the collection to load data from
        :return: DataFrame containing the loaded MongoDB data
        """
        return (
            self.spark.read.format("mongo")
            .option("database", db_name)
            .option("collection", collection_name)
            .load()
        )

    def aggregate_hourly_transactions(self, transactions: DataFrame) -> DataFrame:
        """
        Aggregate transactions by hour, calculating volume and weighted average price
        Note: this is helper function, you do not have to
        """
        return transactions.groupBy(
            date_trunc("hour", "datetime").alias("datetime"), "ticker"
        ).agg(
            sum("shares").alias("volume"),
            (sum(col("shares") * col("price")) / sum("shares")).alias(
                "avg_stock_price"
            ),
        )

        # Function breakdown:
        # 1. Groups transactions by hour and ticker using date_trunc
        # 2. For each group, calculates:
        #    - Total volume: sum of all shares traded
        #    - Weighted average price: (sum of shares * price) / total shares
        # 3. Returns DataFrame with columns: datetime, ticker, volume, avg_stock_price

    def compute_hourly_transactions(
        self, mongo_df: DataFrame, market_df: DataFrame, company_df: DataFrame
    ) -> DataFrame:
        """
        Process MongoDB transactions to extract the ticker and aggregate by hour.
        """
        from pyspark.sql.functions import (
            explode,
            col,
            to_timestamp,
            date_trunc,
        )

        # Exploding the 'transactions' column to get each transaction in a separate row
        exploded_df = mongo_df.withColumn("transaction", explode("transactions"))
        exploded_df = exploded_df.select(
            col("timestamp").alias("datetime"),
            col("transaction.ticker").alias("ticker"),
            col("transaction.shares").alias("shares"),
            col("transaction.price").alias("price"),
        )

        # Hourly aggregate:
        processed_df = self.aggregate_hourly_transactions(exploded_df)
        # Combine processed_df with company_df to add the company_name column.
        transactions_df = processed_df.join(company_df, on="ticker", how="left")

        # Ensure market_df has one record per hour by truncating the timestamp
        market_df = market_df.withColumn(
            "hour", date_trunc("hour", to_timestamp(col("timestamp")))
        )
        # Join using the hour column
        transactions_df = transactions_df.join(
            market_df,
            transactions_df["datetime"] == market_df["hour"],
            how="left",
        )
        transactions_df = transactions_df.select(
            col("datetime"),
            col("ticker"),
            col("company_name"),
            col("avg_stock_price"),
            col("volume"),
            col("index_value").alias("market_index"),
        )
        return transactions_df

    def aggregate_hourly(self, df: DataFrame) -> DataFrame:
        """Standardize the hourly data format"""
        df = df.withColumn("avg_stock_price", col("avg_stock_price").cast("double"))
        return df.select(
            date_format(col("datetime"), "yyyy-MM-dd HH:mm:ss").alias("datetime"),
            col("ticker"),
            col("company_name"),
            round(col("avg_stock_price"), 2).alias("avg_price"),
            (col("volume") / 2).alias("volume"),
            col("market_index"),
        ).orderBy("datetime", "ticker")

        # Function breakdown:
        # 1. Standardizes datetime format to "yyyy-MM-dd HH:mm:ss"
        # 2. Selects and renames required columns:
        #    - datetime: formatted timestamp
        #    - ticker: stock symbol
        #    - company_name: name of the company
        #    - avg_stock_price → avg_price: average price for the hour
        #    - volume: total trading volume
        #    - market_index: relevant market index
        # 3. Orders results by datetime and ticker

    def aggregate_daily_data(self, df: DataFrame) -> DataFrame:
        """Standardize the daily data format"""
        from pyspark.sql.functions import date_format, col, bround, avg, sum, round

        return (    
            df.groupBy(
                date_format(col("datetime"), "yyyy-MM-dd").alias("date"),
                "ticker",
                "company_name",
            )
            .agg(
                (bround(avg("avg_stock_price"), 2)).alias("avg_price"),
                (sum("volume") / 2).alias("volume"),
                bround(avg("market_index"), 2).alias("market_index"),
            )
            .orderBy("date", "ticker")
        )

    def aggregate_monthly_data(self, df: DataFrame) -> DataFrame:
        """Standardize the monthly data format"""
        from pyspark.sql.functions import date_format, col, round, avg, sum

        return (
            df.groupBy(
                date_format(col("datetime"), "yyyy-MM").alias("month"),
                "ticker",
                "company_name",
            )
            .agg(
                round(avg("avg_stock_price"), 2).alias("avg_price"),
                (sum("volume") / 2).alias("volume"),
                round(avg("market_index"), 2).alias("market_index"),
            )
            .orderBy("month", "ticker")
        )

    def aggregate_quarterly_data(self, df: DataFrame) -> DataFrame:
        """Standardize the quarterly data format"""
        from pyspark.sql.functions import (
            month,
            year,
            quarter,
            col,
            concat,
            lit,
            round,
            avg,
            sum,
        )

        return (
            df.filter(month(col("datetime")).between(10, 12))
            .groupBy(
                year(col("datetime")).alias("year"),
                "ticker",
                "company_name",
                quarter(col("datetime")).alias("quarter"),
            )
            .agg(
                round(avg("avg_stock_price"), 2).alias("avg_price"),
                (sum("volume") / 2).alias("volume"),
                round(avg("market_index"), 2).alias("market_index"),
            )
            .select(
                concat(col("year"), lit(" Q"), col("quarter")).alias("quarter"),
                "ticker",
                "company_name",
                "avg_price",
                "volume",
                "market_index",
            )
            .orderBy("quarter", "ticker")
        )

    def save_to_csv(self, df: DataFrame, output_path: str, filename: str) -> None:
        """
        Save DataFrame to a single CSV file.

        :param df: DataFrame to save
        :param output_path: Base directory path
        :param filename: Name of the CSV file
        """
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Create full path for the output file
        full_path = os.path.join(output_path, filename)
        print(f"Saving to: {full_path}")  # Debugging output

        # Create a temporary directory in the correct output path
        temp_dir = os.path.join(output_path, "_temp")
        print(f"Temporary directory: {temp_dir}")  # Debugging output

        # Save to temporary directory
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_dir)

        # Find the generated part file
        csv_file = glob.glob(f"{temp_dir}/part-*.csv")[0]

        # Move and rename it to the desired output path
        shutil.move(csv_file, full_path)

        # Clean up - remove the temporary directory
        shutil.rmtree(temp_dir)
