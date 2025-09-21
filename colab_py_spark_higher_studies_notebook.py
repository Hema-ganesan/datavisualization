# Colab-ready PySpark notebook: Higher Studies (International Graduates Employment Dataset)
# -----------------------------------------------------------------------------
# Instructions:
# 1) Open a new Google Colab notebook.
# 2) Copy & paste each cell below into its own Colab cell (cells are separated by lines like '# %%').
# 3) Run cells top-to-bottom. For the dataset, either upload `dataset.csv` using the upload cell
#    or mount your Google Drive and set the path to the file.
# 4) If something errors, copy the error text and share here — I'll help debug.

# %%
# 1) Install Java (if needed) and required Python packages
# (Colab sometimes already has Java; this command is safe to run regardless)
!apt-get update -qq && apt-get install -y openjdk-11-jdk-headless -qq
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'

# Install PySpark and common plotting libs
!pip install -q pyspark plotly pandas matplotlib

# %%
# 2) Option A - Upload the dataset manually (good for quick one-off runs)
# Run this cell, click Choose Files, and upload dataset.csv (the Kaggle file you saved).
from google.colab import files
print("If you prefer to upload dataset.csv from your computer, run this cell and choose the file.")
uploaded = files.upload()
# After uploading, Colab populates `uploaded` with the filename(s).
# We will pick the first uploaded filename as csv_path below if upload was used.

# %%
# 2) Option B - OR mount Google Drive (if you store dataset.csv in Drive)
# Uncomment and use this instead of Option A if your file is in Google Drive.
# from google.colab import drive
# drive.mount('/content/drive')
# csv_path = '/content/drive/MyDrive/dataset.csv'

# %%
# 3) Initialize SparkSession
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('HigherStudiesAnalysis') \
    .getOrCreate()
print('Spark version:', spark.version)

# %%
# 4) Set csv_path depending on upload or Drive.
# If you used the upload cell above, use the uploaded filename; otherwise set the drive path.
import sys
if 'uploaded' in globals() and len(uploaded) > 0:
    csv_path = list(uploaded.keys())[0]
    print('Using uploaded file:', csv_path)
else:
    # Replace with your Drive path if you mounted drive
    csv_path = '/content/dataset.csv'
    print('Using default path (change if needed):', csv_path)

# %%
# 5) Read CSV into a Spark DataFrame (automatic schema inference)
df = spark.read.option('header', True).option('inferSchema', True).csv(csv_path)
print('Rows:', df.count())
print('Columns:', len(df.columns))

# Quick look
df.printSchema()
df.show(10, truncate=False)

# %%
# 6) Utility: helper to find likely column names for your target dimensions
from pyspark.sql.functions import col

def find_column(df, keywords):
    cols = df.columns
    for k in keywords:
        for c in cols:
            if k.lower() in c.lower():
                return c
    return None

country_col = find_column(df, ['country', 'state', 'location', 'nation', 'place'])
level_col   = find_column(df, ['degree', 'level', 'qualification', 'study_level'])
course_col  = find_column(df, ['course', 'field', 'discipline', 'subject', 'program'])
tuition_col = find_column(df, ['tuition', 'fee', 'cost', 'amount', 'price'])
scholar_col = find_column(df, ['scholar', 'grant', 'bursary', 'fund'])
employ_col  = find_column(df, ['employ', 'work', 'job', 'occupation', 'status'])
year_col    = find_column(df, ['year', 'intake', 'admission'])

print('Detected columns (may be None):')
print(' country_col ->', country_col)
print(' level_col   ->', level_col)
print(' course_col  ->', course_col)
print(' tuition_col ->', tuition_col)
print(' scholar_col ->', scholar_col)
print(' employ_col  ->', employ_col)
print(' year_col    ->', year_col)

# %%
# 7) Cast likely numeric columns to double (so we can compute averages, sums).
#    This is conservative: only attempts to cast columns whose names match common patterns.
numeric_candidates = [tuition_col, scholar_col]
numeric_candidates = [c for c in numeric_candidates if c is not None]

for c in numeric_candidates:
    try:
        df = df.withColumn(c, col(c).cast('double'))
        print(f'Cast {c} -> double')
    except Exception as e:
        print('Could not cast', c, e)

# Re-check dtypes
print('\nColumn types after casting:')
print(df.dtypes)

# %%
# 8) Basic cleaning: drop duplicates and show null counts per detected column
orig_count = df.count()
df = df.dropDuplicates()
print(f'Dropped duplicates: {orig_count} -> {df.count()}')

# Show null counts for detected columns
from pyspark.sql.functions import sum as _sum, when
null_summary = {}
for c in [country_col, level_col, course_col, tuition_col, scholar_col, employ_col, year_col]:
    if c:
        cnt = df.filter(col(c).isNull() | (col(c) == '')).count()
        null_summary[c] = cnt
print('Null / empty counts for detected columns:')
print(null_summary)

# %%
# 9) Example derived columns: is_graduate and is_employed (if relevant columns exist)
from pyspark.sql.functions import when

def add_is_graduate(df, level_col):
    if not level_col:
        return df
    return df.withColumn('is_graduate', when(col(level_col).rlike('(?i)grad|post|master|phd|doctoral'), 1).otherwise(0))

def add_is_employed(df, employ_col):
    if not employ_col:
        return df
    return df.withColumn('is_employed', when(col(employ_col).rlike('(?i)yes|employed|working|full-time|part-time|self-employed'), 1).otherwise(0))

if level_col:
    df = add_is_graduate(df, level_col)
    print('Added is_graduate')
if employ_col:
    df = add_is_employed(df, employ_col)
    print('Added is_employed')

# Show a sample after derived cols
df.select(*([c for c in df.columns if c in [country_col, level_col, course_col, tuition_col, scholar_col, employ_col, 'is_graduate', 'is_employed']])).show(10, truncate=False)

# %%
# 10) Aggregations & example analyses (adjust column names if your dataset uses different names)
# Top countries by number of students
if country_col:
    df.groupBy(country_col).count().orderBy('count', ascending=False).show(10)

# Average tuition by country
if country_col and tuition_col:
    df.groupBy(country_col).avg(tuition_col).withColumnRenamed(f'avg({tuition_col})','avg_tuition').orderBy('avg_tuition', ascending=False).show(20)

# Average tuition by course/field
if course_col and tuition_col:
    df.groupBy(course_col).avg(tuition_col).withColumnRenamed(f'avg({tuition_col})','avg_tuition_by_course').orderBy('avg_tuition_by_course', ascending=False).show(20)

# Employment rate by country (if is_employed exists)
if 'is_employed' in df.columns and country_col:
    emp = df.groupBy(country_col).agg({'is_employed':'avg'}).withColumnRenamed('avg(is_employed)','employment_rate').orderBy('employment_rate', ascending=False)
    emp.show(20)

# %%
# 11) Use Spark SQL for flexible queries
df.createOrReplaceTempView('students')
# Example: top 10 countries with avg tuition and employment rate (if available)
sql = 'SELECT'
if country_col:
    sql += f' {country_col} as country,'
sql += f' COUNT(*) as count_students'
if tuition_col:
    sql += f', AVG({tuition_col}) as avg_tuition'
if 'is_employed' in df.columns:
    sql += f', AVG(is_employed) as employment_rate'
sql += f' FROM students'
if country_col:
    sql += f' GROUP BY {country_col} ORDER BY count_students DESC LIMIT 20'

print('Running SQL:\n', sql)
spark.sql(sql).show(20, truncate=False)

# %%
# 12) Visualizations (convert small aggregated results to Pandas)
# Note: Converting the entire dataset to Pandas may be heavy; instead convert only aggregated/sampled data.
import pandas as pd
import plotly.express as px

# Example 1: Bar chart of top countries by student count
if country_col:
    top_countries = df.groupBy(country_col).count().orderBy('count', ascending=False).limit(15).toPandas()
    fig = px.bar(top_countries, x=country_col, y='count', title='Top countries by student count')
    fig.show()

# Example 2: Avg tuition by course
if course_col and tuition_col:
    avg_course = df.groupBy(course_col).avg(tuition_col).withColumnRenamed(f'avg({tuition_col})','avg_tuition').orderBy('avg_tuition', ascending=False).limit(20).toPandas()
    fig = px.bar(avg_course, x=course_col, y='avg_tuition', title='Avg tuition by course/field')
    fig.show()

# Example 3: Employment rate by country
if 'is_employed' in df.columns and country_col:
    emp_pdf = emp.limit(20).toPandas()
    fig = px.bar(emp_pdf, x='country', y='employment_rate', title='Employment rate (avg is_employed) by country')
    fig.show()

# %%
# 13) Export cleaned/processed dataset
# Small datasets: you can convert to pandas and save as a single CSV file
clean_output_path = '/content/cleaned_higher_studies.csv'
try:
    small_pdf = df.limit(100000).toPandas()  # limit to avoid OOM
    small_pdf.to_csv(clean_output_path, index=False)
    print('Saved sample cleaned CSV to', clean_output_path)
except Exception as e:
    print('Could not convert to pandas (dataset might be large). Consider writing Spark output directly to Drive as Parquet/CSV partitions.')
    # Example: write Spark DataFrame as partitioned parquet (good for large data)
    # df.write.mode('overwrite').parquet('/content/drive/MyDrive/cleaned_higher_studies_parquet')

# %%
# 14) Final notes and cleanup
print('Analysis complete. When finished, stop the Spark session if you want to free resources.')
# spark.stop()  # uncomment to stop

# End of notebook
