# Databricks notebook source
#### THAT NOTEBOOK WAS ONLY USED TO RUN STREAMLIT - ITS NOT A REQUIREMENT WITHOUT THE CLUSTER'S RESTRCTIONS#######

# COMMAND ----------

import pandas as pd
from pygments.lexer import combined

# COMMAND ----------

!curl -I http://localhost:8501


# COMMAND ----------

!hostname -I

# COMMAND ----------

!ngrok authtoken 'your auth token'

# COMMAND ----------

!ngrok tunnels


# COMMAND ----------

!pkill ngrok


# COMMAND ----------

from pyngrok import ngrok
public_url = ngrok.connect(8502)
print("Public URL:", public_url)
