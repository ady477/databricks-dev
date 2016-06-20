# Databricks notebook source exported at Mon, 20 Jun 2016 20:11:43 UTC
# MAGIC %md
# MAGIC ## Issues at a given time
# MAGIC  Author: ** Aditya Tiwari **

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT
# MAGIC   iss.id AS IssueID,
# MAGIC   pro.name AS ProjectName, 
# MAGIC   iss.created_at AS CreatedAt,
# MAGIC   iss.closed_at AS ClosedAt
# MAGIC FROM base_field_issues iss 
# MAGIC   JOIN base_field_projects pro ON iss.project_id = pro.id 
# MAGIC   JOIN base_field_customizable_categories cust ON cust.id = iss.issue_type_id 
# MAGIC   JOIN base_field_categories cat ON cat.project_id = pro.id AND cat.id = cust.category_id 
# MAGIC WHERE 
# MAGIC   iss.project_id = "5e9f4b11-120f-11e4-9511-027465b920a7"
# MAGIC   AND cat.name != "Safety"
# MAGIC   AND iss.created_at >= "2015-01-01"
# MAGIC   AND lower(pro.name) NOT LIKE "%training%"
# MAGIC   AND lower(pro.name) NOT LIKE "%demo%"
# MAGIC   AND cat.name = "QA/QC"

# COMMAND ----------

timeline_freq = 'M' #'D' for days, 'W' for week, M' for months, haven't tested for year yet. 
query = '''SELECT DISTINCT
  iss.id AS IssueID,
  pro.name ProjectName, 
  iss.created_at AS CreatedAt,
  iss.closed_at AS ClosedAt
FROM base_field_issues iss 
  JOIN base_field_projects pro ON iss.project_id = pro.id 
  JOIN base_field_customizable_categories cust ON cust.id = iss.issue_type_id 
  JOIN base_field_categories cat ON cat.project_id = pro.id AND cat.id = cust.category_id 
WHERE 
  iss.project_id = "5e9f4b11-120f-11e4-9511-027465b920a7"
  AND cat.name != "Safety"
  AND iss.created_at >= "2015-01-01"
  AND lower(pro.name) NOT LIKE "%training%"
  AND lower(pro.name) NOT LIKE "%demo%"
  AND cat.name = "QA/QC"
  '''

allIssueDF = sqlContext.sql(query)
#allIssueDF.show()


# COMMAND ----------

df = allIssueDF.select(allIssueDF['IssueID'],allIssueDF['CreatedAt'],allIssueDF['ClosedAt'])
df = df.select(df.IssueID, df.CreatedAt.cast("timestamp").alias("CreatedAt"), df.ClosedAt.cast("timestamp").alias("ClosedAt"))
from pyspark.sql.functions import *
df_createTime = df.sort(df.CreatedAt.asc())
df_closedTime = df.sort(df.ClosedAt.desc())
dfo = df.persist()
dfs = df_createTime.persist()
dfe = df_closedTime.persist()
#dfo.show()


# COMMAND ----------

import pandas as pd
timeline_start = dfs.toPandas().head(1).get_values()[0][1]
timeline_end = dfe.toPandas().head(1).get_values()[0][2]
#print timeline_start
#print timeline_end

# COMMAND ----------

#timeline_freq = 'M' #'D' for days, 'W' for week, M' for months, haven't tested for year yet. 
import datetime
#base = datetime.datetime.today()
#numdays = 10
#date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
#print date_list
tl = pd.date_range(start=timeline_start, end=timeline_end, normalize = True, freq=timeline_freq)
ts = pd.Series(tl, index=range(0,len(tl))) #python pandas dataframe [index date]
print (ts)

# COMMAND ----------

dff = ts.to_frame()
dff['no_of_issues'] = 0
dff.columns = ['timeline','no_of_issues']
#faceprint(dff)

# COMMAND ----------

orig = dfs.toPandas()
for i in range(0,len(dff)):
  count = 0
  for j in range(0,len(orig)):
    if((dff['timeline'][i] > orig['CreatedAt'][j]) and ((pd.isnull(orig['ClosedAt'][j])) or (dff['timeline'][i] < orig['ClosedAt'][j]))):
      count += 1
  dff['no_of_issues'][i] = count
print dff

# COMMAND ----------

from time import strftime
dff['timeline'] = dff['timeline'].apply(lambda x: x.strftime('%Y-%m-%d'))
#print(dff)
from pyspark.sql.types import *
schema = StructType([
  StructField("timeline", DateType(), True),
  StructField("no_of_issues", IntegerType(), True)])
sparkDF = sqlContext.createDataFrame(dff)

# COMMAND ----------

display(sparkDF)

# COMMAND ----------

