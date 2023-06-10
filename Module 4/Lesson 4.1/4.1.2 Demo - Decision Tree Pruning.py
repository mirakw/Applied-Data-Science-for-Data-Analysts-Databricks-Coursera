# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Decision Tree Pruning
# MAGIC
# MAGIC **Objective**: *Demonstrate the use of decision tree pruning to prevent overfitting.*
# MAGIC
# MAGIC In this demo, we will walk through the process of tuning hyperparameters to prune decision trees with sklearn.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prepare data
# MAGIC
# MAGIC Remember that one of our project objectives is to predict a customer's daily average number of steps based on their other recorded metrics. Therefore, we are interested in a user-level aggregation. We will use the `adsda.ht_user_metrics_lifestyle` table that we created in the previous demo.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics_lifestyle LIMIT 10

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

X = ht_lifestyle_pd_df[['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'avg_vo2', 'avg_workout_minutes']]

y = ht_lifestyle_pd_df['steps']

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit a base decision tree
# MAGIC
# MAGIC We will start with a baseline model with no hyperparameter tuning.

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the results

# COMMAND ----------

y_train_predicted = dt.predict(X_train)
y_test_predicted = dt.predict(X_test)

print("R2 on training set: ", round(dt.score(X_train, y_train),3))
print("R2 on test set: ", round(dt.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Overfitting and high variance!
# MAGIC
# MAGIC The decision tree is fitting 100% perfectly on the training set, but doesn't do very well on the test set. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tune the decision tree hyperparameters
# MAGIC
# MAGIC Remember some of the hyperparameters that can be tuned in a decision tree to prevent overfitting on the training set.

# COMMAND ----------

# MAGIC %md
# MAGIC **Maximum tree depth**
# MAGIC    - limiting how deep the tree grows (how many levels of splitting)

# COMMAND ----------

dt_depth = DecisionTreeRegressor(max_depth=4)

dt_depth.fit(X_train, y_train)

y_train_predicted = dt_depth.predict(X_train)
y_test_predicted = dt_depth.predict(X_test)

print("R2 on training set: ", round(dt_depth.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_depth.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The model is no longer overfitting to the training data, and has gotten slightly better on the test set, but now it has high bias (not learning the training set very well). Let's try tuning another hyperparameter.

# COMMAND ----------

# MAGIC %md
# MAGIC  **Minimum node size**
# MAGIC    - requiring that each node have a minimum number of data points in order to split it further

# COMMAND ----------

dt_node = DecisionTreeRegressor(max_depth=6, min_samples_split=3)

dt_node.fit(X_train, y_train)

y_train_predicted = dt_node.predict(X_train)
y_test_predicted = dt_node.predict(X_test)

print("R2 on training set: ", round(dt_node.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_node.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC This is getting better, but the bias and variance are still too high and the performance on the test set is pretty far below the training performance. Let's try something else.

# COMMAND ----------

# MAGIC %md
# MAGIC **Minimum leaf size**
# MAGIC   - Requiring at least a certain number of data points in each leaf

# COMMAND ----------

dt_leaf = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=3)

dt_leaf.fit(X_train, y_train)

y_train_predicted = dt_leaf.predict(X_train)
y_test_predicted = dt_leaf.predict(X_test)

print("R2 on training set: ", round(dt_leaf.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_leaf.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC That didn't help much. We'll try one more.

# COMMAND ----------

# MAGIC %md
# MAGIC **Maximum features**
# MAGIC   - maximum number of features to consider at each split
# MAGIC   - introduces randomness

# COMMAND ----------

dt_features = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=3, max_features=3)

dt_features.fit(X_train, y_train)

y_train_predicted = dt_features.predict(X_train)
y_test_predicted = dt_features.predict(X_test)

print("R2 on training set: ", round(dt_features.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_features.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC We have seen that it is difficult to get a decision tree that doesn't have high variance, even with some tuning of the hyperparameters. In the next lesson, we'll learn about a method that combines decision trees to obtain better models with higher predictive power.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>