# Databricks notebook source
# MAGIC %md
# MAGIC Purpose: Create ADB Notebook for Pyspark Interview Questions
# MAGIC Revision History:
# MAGIC | Date | Author | Description | Execution Time | 
# MAGIC |----------|:-------------:|--------------:| --------------:| 
# MAGIC |July 9, 2025|Puneet Sharma| Create ADB Notebook for Pyspark Interview Question| 2 mins 0sec| 
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-1 : While Ingesting customer data from an external source,you notice duplicate entries.How would you remove duplicates and retain only the latest entry based on a timestamp column?

# COMMAND ----------

data1=[("101","2023-12-01",100),("101","2023-12-02",150),("102","2023-12-01",200),("102","2023-12-02",250)]
schema1=["product_id","date","sales"]
df=spark.createDataFrame(data1,schema1)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### We have date column in string format, we need to change that into date column

# COMMAND ----------

df=df.withColumn('date',col('date').cast(DateType()))
#df=df.withColumn('date',to_date(col('date')))

# COMMAND ----------

##### Method 1 using Dense Rank

# COMMAND ----------

df=df.withColumn('Rank',dense_rank().over(Window.partitionBy("product_id").orderBy(col('date').desc())))
df=df.filter(col('Rank')==1)
df=df.select('product_id','date','sales')
df.display()

# COMMAND ----------

##### Method 2 using sorting

# COMMAND ----------

df=df.orderBy('product_id','date',ascending=[1,0]).dropDuplicates(subset=['product_id'])
df.display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-2 : While Processing Data from Multiple Files with Inconsistent Schema,You need to merge them into a single DataFrame. How would you handle this inconsistency in Pyspark?

# COMMAND ----------

df=spark.read.format('parquet')\
    .option('mergeSchema',True)\
        .load('File/Data/datafiles')

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-3 : You are Working with a real-time data pipeline, and you notice missing values in your streaming data column-Category. How would you handle null or missing values in such a scenario?
# MAGIC
# MAGIC #### df_stream=spark.readStream.schema("id INT,value STRING").csv("path/to/stream)

# COMMAND ----------

df=df.fillNa('Category':'N/A')

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-4 : You need to calculate the total number of actions performed by users in a system. How would you calculate the top 5 most active users based on this information?

# COMMAND ----------

data=[("user1",5),("user2",8),("user3",2),("user4",10),("user2",3)]
schema=["user_id","action"]
df=spark.createDataFrame(data,schema)
df.display()

# COMMAND ----------

df=df.groupBy(col('user_id')).agg(sum(col('action')).alias('Total_Actions')).orderBy('Total_Actions',ascending=False)
df.display()
#If you want top 5 most active users just use limit function in the end
# df=df.groupBy(col('user_id')).agg(sum(col('action')).alias('Total_Actions')).orderBy('Total_Actions',ascending=False).limit(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-5 : While Processing the sales Transaction Data,you need to identify the most recent transactions for each customer.How would you approach this task?

# COMMAND ----------

data1=[("cust1","2023-12-01",100),("cust2","2023-12-02",150),("cust1","2023-12-03",200),("cust2","2023-12-04",250)]
schema1=["customer_id","transaction_date","sales"]
df=spark.createDataFrame(data1,schema1)
df.display()

# COMMAND ----------

df=df.withColumn('transaction_date',to_date('transaction_date'))

# COMMAND ----------

df=df.withColumn('Rank',dense_rank().over(Window.partitionBy(col('customer_id')).orderBy(desc('transaction_date'))))
df.select('customer_id','transaction_date','sales').filter(col('rank')==1).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-6 : You need to identify customers who haven't made any purchases in the last 150 days.How would filter such customers?

# COMMAND ----------

data1=[("cust1","2025-12-01"),("cust2","2024-11-20"),("cust1","2023-12-03"),("cust2","2024-11-25")]
schema1=["customer","transaction_date"]
df=spark.createDataFrame(data1,schema1)
df.display()

# COMMAND ----------

df=df.withColumn('transaction_date',to_date('transaction_date'))

# COMMAND ----------

df=df.withColumn('DateDifference',datediff(current_date(),col('transaction_date')))
df.filter(col('DateDifference')>150).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-7 : While analyzing customer reviews,you need to identify the most frequently used words in the feedback.How would you implement this?

# COMMAND ----------

data=[("customer1","the product is great"),("customer2","Great Product, Fast Delivery"),("customer3","Not bad,could be better")]
schema=['customer_id','feedback']
df=spark.createDataFrame(data,schema)
df.display()

# COMMAND ----------

df=df.withColumn('feedback',lower('feedback')).withColumn('feedback',explode(split('feedback',' ')))
df.display()

# COMMAND ----------

df_group=df.groupBy('feedback').agg(count('feedback'))
df_group.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-8 : You need to calculate the cumulative sum of the sales over time for each product. How would you approach this?

# COMMAND ----------

data1=[("product1","2023-12-01",100),("product2","2023-12-02",200),("product1","2023-12-03",150),("product3","2023-12-04",250)]
schema1=["product_id","date","sales"]
df=spark.createDataFrame(data1,schema1)
df.display()

# COMMAND ----------

df=df.withColumn('cumsum',sum('sales').over(Window.partitionBy('product_id').orderBy('date')))
df=df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-9 : While Preparing a data pipeline you notice some duplicate rows in a dataset. How would you remove the duplicates without affecting the original order?

# COMMAND ----------

data=[('John',25),('Jane',20),('John',25),('Alice',22)]
schema=['name','age']
df=spark.createDataFrame(data,schema)
df.display()

# COMMAND ----------

df=df.withColumn('rank',row_number().over(Window.partitionBy('name').orderBy('age'))).filter(col('rank')==1)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-10 :You are working with user activity data and need to calculate the average session duration per user. How would you implement this? 

# COMMAND ----------

data1=[("user1","2023-12-01",50),("user1","2023-12-02",60),("user2","2023-12-01",45),("user2","2023-12-03",75)]
schema1=["user_id","session_date","duration"]
df=spark.createDataFrame(data1,schema1)
df.display()

# COMMAND ----------

df=df.groupBy('user_id').agg(avg(col('duration'))).alias('Averagesession')
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-11 : While Analysing the sales data,you need to find the product with the highest sales for each month. How would you accomplish this?

# COMMAND ----------

data1=[("product1","2023-12-01",100),("product2","2023-12-01",150),("product1","2023-12-02",200),("product2","2023-12-02",250)]
schema1=["product_id","date","sales"]
df=spark.createDataFrame(data1,schema1)
df.display()

# COMMAND ----------

df=df.withColumn('date',to_date(col('date')))
df=df.withColumn('Month',month(col('date')))
df=df.groupBy('product_id','Month').agg(sum('sales').alias('Total_Sales'))
df=df.withColumn('Rank',dense_rank().over(Window.partitionBy('Month').orderBy(col('Total_Sales').desc()))).filter(col('Rank')==1)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-12 : You are working with a large Delta Table that is frequently updated by multiple users. The data is stored in partitions, and sometimes updates can cause inconsistent reads due to reads due to concurrent transactions. How would you ensure ACID compliance and avoid data corruption in Pyspark?

# COMMAND ----------

df=spark.read.format('parquet').load('path') # This is the source table

from delta.tables import DeltaTable

delta_tbl=DeltaTable.forPath('path')

delta_tbl.alias('target').merge(df.alias('source'),"source.id==target.id")\
                        .whenNotMatchedInsertAll()\
                        .whenNotMatchedUpdateAll()\
                        .execute()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-13 : You need to process a large dataset stored in PARQUET format and ensure that all column have the right schema(ALMOST). How would you do this?

# COMMAND ----------

df=spark.read.format('parquet').option('inferSchema',True).load('path')

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-14 : You are reading a CSV file and need to handle corrupt records gracefully by skipping them. How would you configure this in Pyspark?

# COMMAND ----------

df=spark.read.format('csv')\.
    option("mode","DROPMALFORMED")\.
    load("Staging Location")

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-15 : You have a dataset containing the names of employees and their departments. You need to find the department with the most employees ?

# COMMAND ----------

data = [("Alice", "HR"), ("Bob", "Finance"), ("Charlie", "HR"), ("David", "Engineering"), ("Eve", "Finance")]
columns = ["employee_name", "department"]
df = spark.createDataFrame(data, columns)
df.display()

# COMMAND ----------

df=df.groupBy('department').agg(count('employee_name').alias('EmployeeCount')).sort('EmployeeCount',ascending=False)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-16 : While processing sales data, you need to classify each transaction as either 'High' or 'Low' based on its amount. How would you achieve this using a when condition ?

# COMMAND ----------

data = [("product1", 100), ("product2", 300), ("product3", 50)]
columns = ["product_id", "sales"]
df = spark.createDataFrame(data, columns)
df.display()

# COMMAND ----------

df=df.withColumn('TransactionLevel',when(col('sales')>50,'High').otherwise('Low'))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-17 : While analyzing a large dataset, you need to create a new column that holds a timestamp of when the record was processed. How would you implement this and what can be the best USE CASE ?

# COMMAND ----------

data = [("product1", 100), ("product2", 200), ("product3", 300)]
columns = ["product_id", "sales"]
df = spark.createDataFrame(data, columns)
df.display()

# COMMAND ----------

df = df.withColumn("processed_time",current_timestamp())
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-18 : You need to register this PySpark DataFrame as a temporary SQL object and run a query on it. How would you achieve this?

# COMMAND ----------

data = [("product1", 100), ("product2", 200), ("product3", 300)]
columns = ["product_id", "sales"]
df = spark.createDataFrame(data, columns)
df.display()

# COMMAND ----------

df.createOrReplaceTempView("tempsqldf")

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * FROM tempsqldf WHERE product_id = 'product1';

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-19 : You need to register this PySpark DataFrame as a temporary SQL object and run a query on it (FROM DIFFERENT NOTEBOOKS AS WELL)?

# COMMAND ----------

df.createOrReplaceGlobalTempView("globalview")

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * FROM global_temp.globalview ;

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-20 : You need to query data from a PySpark DataFrame using SQL, but the data includes a nested structure. How would you flatten the data for easier querying?

# COMMAND ----------

 data = [("product1", {"price": 100, "quantity": 2}), ("product2", {"price": 200, "quantity":3})]
 columns = ["product_id", "product_info"]
 df = spark.createDataFrame(data, columns)
df.display()

# COMMAND ----------

df.select("product_id","product_info.price","product_info.quantity").createOrReplaceTempView("flatview")

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * FROM flatview 

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-21 : While reading data from Parquet, you need to optimize performance by partitioning the data based on a column. How would you implement this?

# COMMAND ----------

df.write.format("parquet").mode("append").partitionBy("category").save("location")
 

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-22 :  You are working with a large dataset in Parquet format and need to ensure that the data is written in an optimized manner with proper compression. How would you accomplish this?

# COMMAND ----------

df.write.format("parquet").option("compression","snappy")
 

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### QUESTION-23 : Your company uses a large-scale data pipeline that reads from Delta tables and processes data using complex aggregations. However, performance is becoming an issue due to the growing dataset size. How would you optimize the performance of the pipeline?

# COMMAND ----------

# MAGIC  %sql
# MAGIC
# MAGIC  OPTIMIZE tabledelta ZORDER BY ('order_date')
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-24 : You are processing sales data. Group by product categories and create a list of all product names in each category?

# COMMAND ----------

data = [("Electronics", "Laptop"), ("Electronics", "Smartphone"), ("Furniture", "Chair"), ("Furniture", "table")]

columns = ["category", "product"]
df = spark.createDataFrame(data, columns)
df.display()

# COMMAND ----------

df = df.groupBy('category').agg(collect_list('product').alias('products'))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-25 : You are analyzing orders. Group by customer IDs and list all unique product IDs each customer purchased ?

# COMMAND ----------

 data = [(101, "P001"), (101, "P002"), (102, "P001"), (101, "P001")]

 columns = ["customer_id", "product_id"]
df = spark.createDataFrame(data, columns)
df.display()

# COMMAND ----------

df = df.groupBy('customer_id').agg(collect_set('product_id').alias('unique_products'))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-26 : For customer records, combine first and last names only if the email address exists ?

# COMMAND ----------

 data = [("John", "Doe", "john.doe@example.com"), ("Jane", "Smith", None)]
 columns = ["first_name", "last_name", "email"]
df = spark.createDataFrame(data, columns)
df.display()

# COMMAND ----------

df = df.withColumn("fullname",when(col('email').isNotNull(), concat_ws("-",col('first_name'),col('last_name'))).otherwise
(None))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-27 : You have a DataFrame containing customer IDs and a list of their purchased product IDs. Calculate the number of products each customer has purchased ?

# COMMAND ----------

 data = [
  (1, ["prod1", "prod2", "prod3"]),
  (2, ["prod4"]),
  (3, ["prod5", "prod6"]),
  ]
myschema = "customer_id INT, product_ids array<STRING>"
df = spark.createDataFrame(data, myschema)
df.display()

# COMMAND ----------

df = df.withColumn("number_of_products", size(col('product_ids')))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-28 : You have employee IDs of varying lengths. Ensure all IDs are 6 characters long by padding with leading zeroes ?

# COMMAND ----------

 data = [
  ("1",),
  ("123",),
  ("4567",),
  ]
schema = ["employee_id"]
df = spark.createDataFrame(data, schema)
df.display()

# COMMAND ----------

df = df.withColumn("employee_id",lpad(col('employee_id'),6,"0"))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-29 : You need to validate phone numbers by checking if they start with "91" ?

# COMMAND ----------

 data = [
  ("911234567890",),("811234567890",),("912345678901",),
 ]
schema = ["phone_number"]
df = spark.createDataFrame(data, schema)
df.display()

# COMMAND ----------

df.filter(col('phone_number').startswith('91')).display()

# COMMAND ----------

df.filter(substring(col('phone_number'),1,2) == "91").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-30 : You have a dataset with courses taken by students. Calculate the average number of courses per student ?

# COMMAND ----------

 data = [
    (1, ["Math", "Science"]),
    (2, ["History"]),
    (3,["Art", "PE","Biology"]),

 ]
schema = ["student_id", "courses"]
df = spark.createDataFrame(data, schema)
df.display()

# COMMAND ----------

df = df.withColumn("course_size",size('courses')).groupBy().agg(avg('course_size'))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-31 : You have a dataset with primary and secondary contact numbers. Use the primary number if available; otherwise, use the secondary number ?

# COMMAND ----------

 data = [
    (None , "1234567890"),
    ("9876543210", None),
    ("7894561230","4567891230"),
 ]
schema = ["primary_contact", "secondary_contact"]
df = spark.createDataFrame(data, schema)
df.display()

# COMMAND ----------

df=df.withColumn('PhoneNumber',when(col('primary_contact').isNotNull(),col('primary_contact')).otherwise(col('secondary_contact')))
df.display()

# COMMAND ----------

df = df.withColumn("contact", coalesce(col('primary_contact'),col('secondary_contact')))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### QUESTION-32 : You are categorizing product codes based on their lengths. If the length is 5, label it as "Standard"; otherwise, label it as "Custom" ?

# COMMAND ----------

 data = [
    ("prod1" ,),
    ("prd234",),
    ("pr9876",),
 ]
schema = ["product_code"]
df = spark.createDataFrame(data, schema)
df.display()

# COMMAND ----------

df = df.withColumn("Code_Flag",when(length(col('product_code'))==5,"Standard").otherwise("Custom"))
df.display()
