# Both XGB and DNN are performed simultaneously
# parameters : XGB - param_grid, n_jobs, DNN - max_epochs, lrn_rate , Main - # of iter
## for the integration of various kinds of features
# refer to https://tutorials.pytorch.kr/beginner/saving_loading_models.html for model save and load
# refer to https://quokkas.tistory.com/entry/pytorch%EC%97%90%EC%84%9C-EarlyStop-%EC%9D%B4%EC%9A%A9%ED%95%98%EA%B8%B0 for early-stopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as T
import sys
from scipy.stats import randint as sp_randint
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, auc, f1_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import timeit
from pickle import dump
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

################### Functions ######################

def list2file(lst, fname):
    f = open(fname, "w")
    for feat in lst:
        f.write(feat+"\n")
    f.close()

def scaler(X_train, X_test):
    sc_X = MinMaxScaler(feature_range=(-1, 1))
    array_train = sc_X.fit_transform(X_train)
    array_test = sc_X.transform(X_test)
    X_train_sc = pd.DataFrame(array_train, index=X_train.index, columns=X_train.columns)
    X_test_sc = pd.DataFrame(array_test, index=X_test.index, columns=X_test.columns)
    return X_train_sc, X_test_sc, sc_X

def scaler_final(X_train):
    sc_X = MinMaxScaler(feature_range=(-1, 1))
    array_train = sc_X.fit_transform(X_train)
    X_train_sc = pd.DataFrame(array_train, index=X_train.index, columns=X_train.columns)
    return X_train_sc, sc_X

### XGB
def XGB_main(X, y):
	xgb = XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False)
	clf = RandomizedSearchCV(estimator = xgb, param_distributions = xgb_param_grid, n_iter = 100, cv = stfold, n_jobs = -1)
	clf.fit(X, y)
	cv_results_ = clf.cv_results_
	best_params_ = clf.best_params_
	best_score_ = clf.best_score_
	best_estimator_ = clf.best_estimator_
	return cv_results_, best_params_, best_score_, best_estimator_

### SVM
def SVM_main(X, y):
	clf = RandomizedSearchCV(estimator = SVC(), param_distributions = svm_param_grid, n_iter = 1, cv = stfold, n_jobs = 10, scoring="roc_auc")
	clf.fit(X, y)
	cv_results_ = clf.cv_results_
	best_params_ = clf.best_params_
	best_score_ = clf.best_score_
	best_estimator_ = clf.best_estimator_
	return cv_results_, best_params_, best_score_, best_estimator_

### RF
def RF_main(X, y):
	clf = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = rf_param_grid, n_iter = 100, cv = stfold, n_jobs = -1, scoring="roc_auc")
	clf.fit(X, y)
	cv_results_ = clf.cv_results_
	best_params_ = clf.best_params_
	best_score_ = clf.best_score_
	best_estimator_ = clf.best_estimator_
	return cv_results_, best_params_, best_score_, best_estimator_

### DNN
class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(num_feat, 4) # Initial : 4-(8-8)-1
        self.hid2 = T.nn.Linear(4, 8)
        self.hid3 = T.nn.Linear(8, 8)
        self.oupt = T.nn.Linear(8, 1)
        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.hid3.weight)
        T.nn.init.zeros_(self.hid3.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)
    def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = T.tanh(self.hid2(z))
        z = T.tanh(self.hid3(z))
        z = T.sigmoid(self.oupt(z))
        return z

class Batcher:
    def __init__(self, num_items, batch_size, seed=0):
        self.indices = np.arange(num_items)
        self.num_items = num_items
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.rnd.shuffle(self.indices)
        self.ptr = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration # exit calling for-loop
        else:
            result = self.indices[self.ptr:self.ptr+self.batch_size]
            self.ptr += self.batch_size
            return result

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        T.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def AUC(model, data_x, data_y):
    X = T.Tensor(data_x)
    Y = T.ByteTensor(data_y) # a Tensor of 0s and 1s
    oupt = model(X)          # a Tensor of floats
    score = average_precision_score(Y.numpy(), oupt.detach().numpy())
    return score

def DNN_main(X_train, X_test, y_train, y_test):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    # change the data format
    train_x = X_train.to_numpy()
    train_y = y_train.to_numpy().reshape(-1,1)
    test_x = X_test.to_numpy()
    test_y = y_test.to_numpy().reshape(-1,1)
    # building a model
    net = Net()
    net = net.train()
    loss_func = T.nn.BCELoss()
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)
    n_items = len(train_x)
    bat_size = round(n_items/5)
    batcher = Batcher(n_items, bat_size)
    # training
    for epoch in range(1, max_epochs+1):
        print("\nTraining START")
        for curr_bat in batcher:
            X = T.Tensor(train_x[curr_bat])
            Y = T.Tensor(train_y[curr_bat])
            optimizer.zero_grad()
            oupt = net(X)
            loss_obj = loss_func(oupt, Y)
            loss_obj.backward()
            optimizer.step()
            train_losses.append(loss_obj.item())
        print("Training COMPLETE")
        Auc = AUC(net, train_x, train_y)
        print("train ap score = %0.2f%%" % Auc)
        # evaluation of held-out test set
        net = net.eval()
        X_test = T.Tensor(test_x)
        Y_test = T.Tensor(test_y)
        oupt_test = net(X_test)
        loss_ = loss_func(oupt_test, Y_test)
        valid_losses.append(loss_.item())
        auc = AUC(net, test_x, test_y)
        print("test ap score on = %0.2f%%" % auc)
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(max_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{max_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        early_stopping(valid_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    net.load_state_dict(T.load('checkpoint.pt'))
    return net, avg_train_losses, avg_valid_losses, auc

def DNN_final_train(X_train, y_train):
    train_losses = []
    avg_train_losses = []
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    # change the data format
    train_x = X_train.to_numpy()
    train_y = y_train.to_numpy().reshape(-1,1)
    # building a model
    net = Net()
    net = net.train()
    loss_func = T.nn.BCELoss()
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)
    n_items = len(train_x)
    bat_size = round(n_items/5)
    batcher = Batcher(n_items, bat_size)
    # training
    for epoch in range(1, max_epochs+1):
        print("\nTraining START")
        for curr_bat in batcher:
            X = T.Tensor(train_x[curr_bat])
            Y = T.Tensor(train_y[curr_bat])
            optimizer.zero_grad()
            oupt = net(X)
            loss_obj = loss_func(oupt, Y)
            loss_obj.backward()
            optimizer.step()
            train_losses.append(loss_obj.item())
        print("Training COMPLETE")
        Auc = AUC(net, train_x, train_y)
        print("train ap score = %0.2f%%" % Auc)
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        epoch_len = len(str(max_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{max_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f}')
        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
    return net, avg_train_losses, Auc

###########################################
#### Training data Processing ####
start = timeit.default_timer()
print("\n"*5)
print("#"*10+" PROCESS START "+"#"*10)

type_ = sys.argv[1] # ANY, SKIN, ENDO etc.
labmd = sys.argv[2] # all, within, onlyCtr
cores = int(sys.argv[3])
n_iter = int(sys.argv[4])

## Inputs
f_gt = "/home/changhwan/amc/08.Analysis/33.final/02.wes/VQSR_nonsyn_608_header.txt"
f_cli = "/home/changhwan/amc/01.Clinial_info/ver15_221211/Processed_metadata_feat"
f_drug = "/home/changhwan/amc/01.Clinial_info/ver15_221211/Processed_metadata_feat_drug2" ## Mono
f_lab = "/home/changhwan/amc/01.Clinial_info/ver15_221211/Processed_metadata_lab_final"
f_hla = "/home/changhwan/jh_irAE/01_data/Asan_normal_HLA.txt"
f_cnv = "/home/changhwan/amc/08.Analysis/33.final/11.cnv/02_Asan_cnv_exon_header2.txt"

## Dataframes for significant feature indexes
f_snp_cm = "/home/changhwan/amc/08.Analysis/33.final/02.wes/06.maf_lr_onlyCtr_cm/06_maf_lr_onlyCtr_%s_filt_res_sorted_rsid_proc_clumped_sorted.bed"%(type_) # for snpid
#f_snp_cm = "/home/omics/DATA1/changhwan/Asan_irAE/08.Analysis/33.final/02.wes/06.maf_lr_onlyCtr_cm/13.ANY_fdr/06_maf_lr_onlyCtr_ANY_filt_res_sorted_rsid_proc_clumped_sorted.bed"
f_res_cli = "/home/changhwan/amc/08.Analysis/33.final/00.clinical/05.analysis_table1_p001.res"
f_res_hla = "/home/changhwan/amc/08.Analysis/33.final/01.hla/05_re_lr_sort_p001.res"
#f_res_cm = "/home/changhwan/amc/08.Analysis/33.final/02.wes/06.maf_lr_onlyCtr_cm/09.merge_res_lr_ap_final.txt"
f_res_cnv = "/home/changhwan/amc/08.Analysis/33.final/11.cnv/03_cnv_lr_onlyCtr2/03_%s_p001_gene.tsv"%(type_)
#f_res_cnv = "/home/changhwan/amc/08.Analysis/33.final/11.cnv/03_cnv_lr_onlyCtr2/01_ANY_fdr_filt2.res"

#### Processing cov df ####
df_cli = pd.read_csv(f_cli,sep="\t",index_col=0)
df_drug = pd.read_csv(f_drug, sep="\t", index_col=0)
df_cov = pd.merge(df_cli, df_drug["MONO"], left_index=True, right_index=True)
df_cov.Sex = df_cov.Sex - 1

df_res_cli = pd.read_csv(f_res_cli, sep="\t")
lst_cli = df_res_cli.loc[df_res_cli["lab"]==type_,"feat"].to_list()
lst_cli.append("Sex")
lst_cli.append("Age")

df_cli_sub = df_cov.loc[df_cov["MONO"]==1,lst_cli]

#### Processing LAB df ####
df_lab = pd.read_csv(f_lab, sep="\t", index_col=0)
df_lab_sub = df_lab[type_]

#### Processing HLA dataframe ####
df_hla = pd.read_csv(f_hla, sep="\t", index_col=0, header=None)
lst_sm = list(df_hla.index)
lst_new = [x.split("D-")[1] for x in lst_sm]
df_hla.index = lst_new
hla_inc = [1,2,3,4,5,6,7,8,11,12,15,16] # A,B,C,DRB1, DQB1, DPB1
df_hla = df_hla.loc[:,hla_inc]
df_hla_stacked = df_hla.stack().str.get_dummies().sum(level=0)

df_res_hla = pd.read_csv(f_res_hla, sep="\t")
set_hla = set(df_res_hla.loc[df_res_hla["lab"]==type_,"hla"].to_list())

df_hla_sub = df_hla_stacked[set_hla]

#### Processing CNV dataframe ####
df_cnv_tmp = pd.read_csv(f_cnv, sep="\t",index_col=0)
df_cnv = df_cnv_tmp.drop("gene",axis=1)
## change the format of sample name
lst_sm = list(df_cnv.columns)
lst_new = [x.split("D-")[1] for x in lst_sm]
df_cnv.columns = lst_new

df_res_cnv = pd.read_csv(f_res_cnv, sep="\t")
set_cnv = set(df_res_cnv["cnvID"].to_list())
df_cnv_sub = df_cnv.T[set_cnv]

#### Processing SNP df ####
df_gt = pd.read_csv(f_gt, sep="\t", index_col=0)
df_cm = pd.read_csv(f_snp_cm, sep="\t")
#df_res_cm = pd.read_csv(f_res_cm, sep="\t")

## change the format of sample name
lst_sm = list(df_gt.columns)
lst_new = [x.split("D-")[1] for x in lst_sm]
df_gt.columns = lst_new

## change index name of df_snp to CHROM_POS_REF_ALT
df_cm[['start','end']] = df_cm[['start','end']].astype(str)
df_cm.index = df_cm[['chr','start','ref','alt']].agg('_'.join,axis=1) ## agg !!
df_cm.index.name = "CHROM_POS_REF_ALT"
lst_cm = list(df_cm.index)

## check the nSNP with highest auc_test score
#nsnp_cm = int(df_res_cm.loc[df_res_cm["label"]==type_,"nSNP_inf"])
#lst_cm_sub = lst_cm[:nsnp_cm]

df_cm_sub = df_gt.loc[lst_cm,:].T


#### Merge various kinds of feature dataframes

#dfs = [df_cli_sub, df_hla_sub, df_cnv_sub, df_cm_sub, df_lab_sub]
dfs = [df_cli_sub, df_hla_sub, df_cm_sub, df_lab_sub]
df_merge_tmp = pd.concat(dfs, join="inner", axis=1, sort=True)
df_merge = df_merge_tmp.loc[:,df_merge_tmp.describe().loc["count",:]>565].dropna() # 3% cut - Protein

if labmd=="all":
	df_merge[type_].replace(2,0, inplace=True)
elif labmd=="within":
	df_merge[type_].replace(0, np.nan, inplace=True)
	df_merge[type_].replace(2,0, inplace=True)
else: #"onlyCtr"
	pass

df_merge_sub = df_merge.loc[df_merge[type_]!=2,:].dropna()


################################ Main #########################
#### Training and cross-validation... optimization process ####

df_X = df_merge_sub.drop(type_, axis=1)
df_y = df_merge_sub[type_]

df_X_sc, sc_final = scaler_final(df_X)

lst_feat = list(df_X.columns)
num_feat = df_X.shape[1]

#### XGBoost ####
xgb_param_grid = {
        "min_child_weight":[0,1,2],
        "gamma":[0,1],
        "n_estimators" : [int(x) for x in np.linspace(start=200, stop=1000, num=3)],
        "max_depth" : [int(x) for x in np.linspace(start=4, stop=10, num=4)],
        "learning_rate" :[0.001, 0.01, 0.1],
        "alpha": [0,1],
        "lambda" : [1,2],
        "eta": [0.01, 0.1, 0.2],
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.5, 1.0]}

stfold = StratifiedKFold(n_splits=5)

cv_results_xgb, best_params_xgb, best_score_xgb, best_estimator_xgb = XGB_main(df_X, df_y)

saved_model_xgb = pickle.dumps(best_estimator_xgb)
#xgb_from_pickle = pickle.loads(saved_model_xgb)
#xgb_from_pickle.predict(X)
f_xgb = open("results3_xgb.tsv","w")
f_xgb.write("cv_results:\n%s\n\nbest_params:\n%s\n\nbest_score:\n%s"%(cv_results_xgb, best_params_xgb, best_score_xgb))
f_xgb.close()
joblib.dump(best_estimator_xgb, 'best_estimator3_xgb.pkl')
#xgb_from_joblib = joblib.load('best_estimator_xgb.pkl') 
#xgb_from_joblib.predict(X)
#average_precision_score(df_y, xgb_from_joblib.predict(df_X))

print("XGB_Training_Done!!\n", flush=True)

#### SVM ####
svm_param_grid = {
	'C': [0.001, 0.01, 0.1, 1, 10, 100], 
	'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
	'kernel': ['rbf', 'poly', 'sigmoid']}

cv_results_svm, best_params_svm, best_score_svm, best_estimator_svm = SVM_main(df_X_sc, df_y)
f_svm = open("results3_svm.tsv","w")
f_svm.write("cv_results:\n%s\n\nbest_params:\n%s\n\nbest_score:\n%s"%(cv_results_svm, best_params_svm, best_score_svm))
f_svm.close()
joblib.dump(best_estimator_svm, 'best_estimator3_svm.pkl')

print("SVM_Training_Done!!\n", flush=True)

#### RF ####
rf_param_grid = {
	'bootstrap': [True, False],
	'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
	'max_features': ['auto', 'sqrt'],
	'min_samples_leaf': [1, 2, 4],
	'min_samples_split': [2, 5, 10],
	'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

cv_results_rf, best_params_rf, best_score_rf, best_estimator_rf = RF_main(df_X, df_y)
f_rf = open("results3_rf.tsv","w")
f_rf.write("cv_results:\n%s\n\nbest_params:\n%s\n\nbest_score:\n%s"%(cv_results_rf, best_params_rf, best_score_rf))
f_rf.close()
joblib.dump(best_estimator_rf, 'best_estimator3_rf.pkl')

print("RF_Training_Done!!\n", flush=True)


#### DNN ####
lrn_rate = 0.001
max_epochs = 20
patience = 5

seed = 8
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2,\
        random_state=seed, stratify=df_y)
X_train_sc, X_test_sc, sc_X = scaler(X_train, X_test)

# Training final model with whole dataset ####
df_X_sc, sc_final = scaler_final(df_X)
model, train_loss_final, auc_final = DNN_final_train(df_X_sc, df_y)


#### Predict new test set #### (n=169)
indir = "/home/changhwan/amc/08.Analysis/34.revision/00_inputs/"
f2_cnv = indir+"02_Asan169_cnv_exon_header2.txt"
f2_hla = indir+"Asan169_normal_HLA.txt"
f2_gt = indir+"VQSR_nonsyn_169_header.txt"
#f2_cli = "/home/changhwan/amc/01.Clinial_info/ver14_220419/Processed_metadata_feat"
f2_cli = "/home/changhwan/amc/01.Clinial_info/ver15_221211/Processed_metadata_feat"
#f2_lab = "/home/changhwan/amc/01.Clinial_info/ver14_220419/Processed_metadata_lab_final"
f2_lab = "/home/changhwan/amc/01.Clinial_info/ver15_221211/Processed_metadata_lab_final"

# snp
df2_gt_tmp = pd.read_csv(f2_gt, sep="\t", index_col=0)
lst_sm = list(df2_gt_tmp.columns)
lst_new = [x.split("D-")[1] for x in lst_sm]
df2_gt_tmp.columns = lst_new
df2_gt = df2_gt_tmp.T

# cnv
df2_cnv_tmp = pd.read_csv(f2_cnv, sep="\t",index_col=0)
df2_cnv_tmp = df2_cnv_tmp.drop("gene",axis=1)
# change the format of sample name
lst_sm = list(df2_cnv_tmp.columns)
lst_new = [x.split("D-")[1] for x in lst_sm]
df2_cnv_tmp.columns = lst_new
df2_cnv = df2_cnv_tmp.T
lst_sm2 = list(df2_cnv.index)

# lab
df2_lab = pd.read_csv(f2_lab, sep="\t", index_col=0)
lst_sm_lab = list(df2_lab.index)
lst_final = list(set(lst_sm2) & set(lst_sm_lab))
df2_lab_sub = df2_lab.loc[lst_final, type_]

# hla
df2_hla = pd.read_csv(f2_hla, sep="\t", index_col=0, header=None)
lst_sm = list(df2_hla.index)
lst_new = [x.split("D-")[1] for x in lst_sm]
df2_hla.index = lst_new
#hla_inc = [3,4,7,8,11,12,15,16] # B,C,DRB1,DQB1,DPB1 => B,DRB1, DQB1, DPB1
hla_inc = [1,2,3,4,5,6,7,8,11,12,15,16] # A,B,C,DRB1, DQB1, DPB1
df2_hla = df2_hla.loc[:,hla_inc]
df2_hla_stacked = df2_hla.stack().str.get_dummies().sum(level=0)

# cli
df2_cli = pd.read_csv(f2_cli, sep="\t", index_col=0)
df2_cov = pd.merge(df2_cli, df_drug["MONO"], left_index=True, right_index=True)
df2_cov.Sex = df2_cov.Sex - 1

# merge
#dfs2 = [df2_cov, df2_hla_stacked, df2_cnv, df2_gt, df2_lab_sub]
dfs2 = [df2_cov, df2_hla_stacked, df2_gt, df2_lab_sub]
df2_merge_tmp = pd.concat(dfs2, join="inner", axis=1, sort=True)
df2_merge = df2_merge_tmp.loc[df2_merge_tmp["MONO"]==1,:]
df2_test = df2_merge[lst_feat].dropna()
lst_test = list(df2_test.index)
test_y = df2_merge.loc[lst_test, type_]
#df2_test_tmp = pd.merge(df2_test, test_y, how="inner", left_index=True, right_index=True)
#df_total = pd.concat([df_merge_sub, df2_test_tmp])

array_test = sc_final.transform(df2_test)
df2_test_sc = pd.DataFrame(array_test, index=df2_test.index, columns=df2_test.columns)

#### prediction

## DNN
inp_X = T.Tensor(df2_test_sc.to_numpy())
model = model.eval()
pred_y = model(inp_X)
ap_dnn = average_precision_score(test_y.to_numpy(), pred_y.detach().numpy())
roc_dnn = roc_auc_score(test_y.to_numpy(), pred_y.detach().numpy())

## XGB
xgb_from_joblib = joblib.load('best_estimator3_xgb.pkl')
pred_y_xgb = xgb_from_joblib.predict(df2_test)
ap_xgb = average_precision_score(test_y, pred_y_xgb)
roc_xgb = roc_auc_score(test_y, pred_y_xgb)

## SVM
svm_from_joblib = joblib.load('best_estimator3_svm.pkl')
pred_y_svm = svm_from_joblib.predict(df2_test_sc)
ap_svm = average_precision_score(test_y, pred_y_svm)
roc_svm = roc_auc_score(test_y, pred_y_svm)

## RF
rf_from_joblib = joblib.load('best_estimator3_rf.pkl')
pred_y_rf = rf_from_joblib.predict(df2_test)
ap_rf = average_precision_score(test_y, pred_y_rf)
roc_rf = roc_auc_score(test_y, pred_y_rf)

# save the result
f_res_test = open("results_test_NC.tsv","w")
f_res_test.write("ap_xgb : %f\nap_svm : %f\nap_rf : %f\nap_dnn : %f\n"%(ap_xgb, ap_svm, ap_rf, ap_dnn))
f_res_test.write("roc_xgb : %f\nroc_svm : %f\nroc_rf : %f\nroc_dnn : %f\n"%(roc_xgb, roc_svm, roc_rf, roc_dnn))
f_res_test.close()
