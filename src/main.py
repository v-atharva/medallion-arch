from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from typing import Dict, Tuple
from data_processor import DataProcessor
from dotenv import load_dotenv
import os
import yaml


def create_spark_session(mysql_connector_path: str, mongodb_uri: str) -> SparkSession:
    """
    Create and return a SparkSession with necessary configurations.

    :param mysql_connector_path: Path to MySQL JDBC connector JAR
    :param mongodb_uri: URI for MongoDB connection
    :return: Configured SparkSession
    """
    return (
        SparkSession.builder.appName("StockDataWarehouse")
        .config("spark.jars", mysql_connector_path)
        .config(
            "spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1"
        )
        .config("spark.mongodb.input.uri", mongodb_uri)
        .getOrCreate()
    )


def main(config: Dict[str, str]) -> Tuple[DataFrame, SparkSession]:
    """
    Main function to process stock data for data warehouse

    :param config: Configuration dictionary containing database and path information
    :return: Tuple of the processed DataFrame and SparkSession
    """
    spark = create_spark_session(config["mysql_connector_path"], config["mongodb_uri"])
    data_processor = DataProcessor(spark)

    # TODO: Implement the following steps:
    # 1. Load data from sources (MySQL and MongoDB)
    # a. MySQL: Company DF - ticker, company_name
    company_df = data_processor.fetch_mysql_data(
        config["mysql_url"],
        config["company_info_table"],
        config["mysql_user"],
        config["mysql_password"],
    )
    print("Company Data:")
    company_df.show(5)

    # b. MySQL: Market_df - timestamp, index_value
    market_df = data_processor.fetch_mysql_data(
        config["mysql_url"],
        config["market_index_table"],
        config["mysql_user"],
        config["mysql_password"],
    )
    print("Market Data:")
    market_df.show(5)

    # c. MongoDB: Mongo_df - _id, timestamp, transaction_id, transactions, user_id
    # (relevant): timestamp, transactions (which can be further exploded)
    mongo_df = data_processor.fetch_mongo_data(
        config["mongo_db"], config["mongo_collection"]
    )
    print("MongoDB Data:")
    mongo_df.show(5)

    # 2. Process hourly transactions
    # Dataframe returned here has the exact format as desired in results.
    # Will use this to aggregate further by hour, quarter, month etc.
    processed_df = data_processor.compute_hourly_transactions(
        mongo_df, market_df, company_df
    )
    processed_df.show(5)

    # 3. Create aggregations (hourly/daily/monthly/quarterly)
    # 3. Create aggregations (hourly/daily/monthly/quarterly)
    print("Hourly data:")
    hourly_df = data_processor.aggregate_hourly(processed_df)
    hourly_df.show(5)

    print("Daily data:")
    daily_df = data_processor.aggregate_daily_data(processed_df)
    daily_df.show(5)

    print("Monthly data:")
    monthly_df = data_processor.aggregate_monthly_data(processed_df)
    monthly_df.show(5)

    print("Quarterly data:")
    quarterly_df = data_processor.aggregate_quarterly_data(processed_df)
    quarterly_df.show(5)

    # 4. Save results to CSV files
    data_processor.save_to_csv(
        hourly_df, config["output_path"], "hourly_stock_data.csv"
    )
    data_processor.save_to_csv(daily_df, config["output_path"], "daily_stock_data.csv")
    data_processor.save_to_csv(
        monthly_df, config["output_path"], "monthly_stock_data.csv"
    )
    data_processor.save_to_csv(
        quarterly_df, config["output_path"], "quarterly_stock_data.csv"
    )
    print("Saved as csv and completed the implementation")

    return None, spark


if __name__ == "__main__":
    # Load configuration
    load_dotenv()

    # Load and process config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        # Replace environment variables
        for key, value in config.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    print(f"Warning: Environment variable {env_var} not found")
                config[key] = env_value or value

    processed_df, spark = main(config)
    spark.stop()
