# Sampel Kode Program
# Proyek Sistem Temu Balik Informasi
# Genap T.A. 2020/2021
# Author : Rizu Jain
# Logistic Regression
# Pointwise Learning to Rank


import pandas as pd
import math

from sklearn.linear_model import LogisticRegression

# Fungsi untuk menyesuaikan kolom 
# pada DataFrame LETOR
# 0 -> Query id
# 1 -> Document id
# 2-47 -> Features
# 48 -> ranking
# 49 -> label
def adjustLETOR(df):
    df[96] = df[0]
    df[0] = df[2]
    df[2] = df[97]
    drop_cols = list(range(1, 96, 2))
    drop_cols.extend(range(97, 104))
    df_adjusted = df.drop(drop_cols, 1)
    df_adjusted.columns = list(range(0, 49))
    df_adjusted[49] = df_adjusted[48] > 0
    df_adjusted.infer_objects()
    df_adjusted[49] = df_adjusted[49].apply(int)
    
    return df_adjusted

# Fungsi untuk menghitung nilai NDCG
# df_pred harus meliputi query_id, document_id, score
def NDCG(df_pred):
    final = df_pred.sort_values([0, 51], ascending=[False, False])
    
    query_id = final[0][0]
    user_rel = []
    
    ndcg = 0
    query_count = 0
    for j in range(final.shape[0]):
        if query_id != final[0][j]:
            # hitung nilai NDCG
            dcg = 0
            for k in range(min(10,len(user_rel))):
                dcg += user_rel[k]/(math.log(k+2))
                
            ideal_rel = sorted(user_rel,reverse=True)
            idcg = 0
            
            for k in range(min(10,len(user_rel))):
                idcg += (ideal_rel[k])/(math.log(k+2))                
            
            if idcg:
                ndcg += (dcg/idcg)
            
            query_id = final[0][j]
            user_rel = [final[48][j]]
            query_count += 1
        else:
            if len(user_rel)<10:
                user_rel.append(final[48][j])
    
    return ndcg/query_count

# Fungsi untuk melakukan learning to rank
def learn_to_rank(Folder):
    
    df_raw_train = pd.read_csv(Folder + "train.txt", " |:", 
                               header=None, engine='python')
    train_df = adjustLETOR(df_raw_train)
    
    df_raw_val = pd.read_csv(Folder + "vali.txt", " |:", 
                             header=None, engine='python')
    val_df = adjustLETOR(df_raw_val)
    
    df_raw_test = pd.read_csv(Folder + "test.txt", " |:", 
                              header=None, engine='python')
    test_df = adjustLETOR(df_raw_test)
    
    X_train = train_df.iloc[:,2:48]
    Y_train = train_df.iloc[:,49]
    
    X_val = val_df.iloc[:,2:48]
    
    X_test = test_df.iloc[:,2:48]
    
    results = []
    
    C = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    
    for hyp in C:
        logreg_class = LogisticRegression(C=hyp, 
                                          max_iter = 500,
                                          class_weight = 'balanced',
                                          solver='lbfgs')
        
        # Bangun model klasifikasi
        logreg_class.fit(X_train, Y_train)
        predictions = logreg_class.predict_proba(X_val)
        val_df[51] = predictions[:,1]
        
        val_ndgc = NDCG(val_df)
        results.append(val_ndgc)
    
    best_hyp = results.index(max(results))
    logreg_class = LogisticRegression(C = C[best_hyp],
                                      max_iter = 500,
                                      class_weight = 'balanced',
                                      solver='lbfgs')
    
    logreg_class.fit(X_train, Y_train)
    predictions = logreg_class.predict_proba(X_test)
    test_df[51] = predictions[:,1]
    test_ndgc = NDCG(test_df)
    
    return test_ndgc

list_ndcg = []
for idx in range(1,6):
    directory = "MQ2008/Fold"+str(idx)+"/"
    dir_ndgc = learn_to_rank(directory)
    list_ndcg.append(dir_ndgc)
    print("NDGC for fold " + str(idx) + " : ", round(dir_ndgc, 2))
    
print(round(sum(list_ndcg) / len(list_ndcg), 2))