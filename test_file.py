import time
from datetime import datetime

# ==== ======= This is the insert part
import matplotlib 
matplotlib.use('Agg')
 # ==== ======= This is the insert part
import matplotlib.pyplot as plt

now = datetime.now()
dt = now.strftime("%d-%m-%Y %H:%M:%S")
print(dt)

time.sleep(5)

now = datetime.now()
dt = now.strftime("%d-%m-%Y %H:%M:%S")
print(dt)