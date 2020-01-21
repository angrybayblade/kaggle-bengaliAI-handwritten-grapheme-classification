import pandas as pd
import time

def load_data(index):
    t = time.time()
    d = pd.read_parquet(f"./train/train_image_data_{index}.parquet")
    print (f"Loaded In {time.time() - t}")
    return d

df = load_data(0)

df_ = load_data(1)
df = pd.concat([df,df_],axis=0)

time.sleep(5)

df_ = load_data(2)
df = pd.concat([df,df_],axis=0)

time.sleep(5)

df_ = load_data(3)
df = pd.concat([df,df_],axis=0)

time.sleep(5)
print (df.shape)
df.to_csv("./train/train_full.csv")

del df,df_,pd,load_data
exit()
