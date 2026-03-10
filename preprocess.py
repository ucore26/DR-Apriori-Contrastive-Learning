import miceforest as mf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

def main():
    sns.set(font='Microsoft YaHei', font_scale=1.5)
    plt.rcParams['font.family'] = ["Microsoft YaHei"]
    plt.rcParams['axes.unicode_minus'] = False

    df = pd.read_excel("./data/data-2022.6.13.xlsx", header=0, dtype=str)
    
    data_missing = df[df.columns[32:45]]
    
    data_missing.rename(columns={'血红蛋白': 'Hemoglobin'}, inplace=True)
    data_missing.rename(columns={'红细胞': 'RBC Count'}, inplace=True)
    data_missing.rename(columns={'白蛋白': 'Albumin'}, inplace=True)
    data_missing.rename(columns={'白细胞': 'CSF WBC Count'}, inplace=True)
    data_missing.rename(columns={'葡萄糖': 'CSF Glucose'}, inplace=True)
    data_missing.rename(columns={'蛋白': 'CSF Protein'}, inplace=True)
    data_missing.rename(columns={'凝血酶原时间': 'Prothrombin Time'}, inplace=True)
    data_missing.rename(columns={'国际标准化的比值': 'INR'}, inplace=True)
    data_missing.rename(columns={'活化部分凝血活酶时间': 'APTT'}, inplace=True)
    data_missing.rename(columns={'凝血酶时间': 'Thrombin Time'}, inplace=True)
    data_missing.rename(columns={'纤维蛋白原': 'Fibrinogen'}, inplace=True)
    data_missing.rename(columns={'D-二聚体': 'D-dimer'}, inplace=True)
    
    cols = ['Hemoglobin', 'RBC Count', 'Albumin', 'CSF WBC Count', 'CSF Glucose', 
            'CSF Protein', 'Prothrombin Time', 'INR', 'APTT', 'Thrombin Time', 
            'Fibrinogen', 'D-dimer']
    data_missing[cols] = data_missing[cols].apply(lambda x: pd.to_numeric(x, errors='ignore'))
    data_missing['术后是否感染'] = pd.to_numeric(data_missing['术后是否感染'], errors='ignore', downcast='integer')
    
    object_cols = data_missing.select_dtypes(include=['object']).columns
    for col in object_cols:
        data_missing[col] = data_missing[col].astype('category')
    
    kernel = mf.ImputationKernel(
        data_missing,
        num_datasets=5,
        random_state=20
    )
    
    kernel.mice(
        iterations=6,
        save_all_iterations=True
    )
    
    completed_data = kernel.complete_data(dataset=3)
    
    for i in range(483):
        if df.iloc[i, 6] != '':
            blood = df.iloc[i, 6].split('/')
            if int(blood[0]) > 140 or int(blood[1]) > 90:
                df.iloc[i, 7] = 2
            else:
                df.iloc[i, 7] = 1
        else:
            df.iloc[i, 7] = 0
    
    df.to_excel("./data/2020.6.13.xlsx", index=False)
    
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = ["Microsoft YaHei"]
    kernel.plot_imputed_distributions(wspace=0.4, hspace=0.6)
    plt.savefig('./data/Figure/imputed_distributions.jpg', dpi=500, bbox_inches='tight')
    
    kernel.plot_mean_convergence(wspace=0.4, hspace=0.6)
    plt.savefig('./data/Figure/mean_convergence.jpg', dpi=500, bbox_inches='tight')
    
    kernel2 = mf.ImputationKernel(
        data_missing,
        datasets=4,
        save_all_iterations=True,
        save_models=1,
        random_state=20,
        mean_match_candidates=7
    )
    
    kernel2.mice(3)
    
    completed_data = kernel2.complete_data(dataset=3)
    
    kernel2.plot_imputed_distributions(wspace=0.4, hspace=0.6)
    plt.savefig('./data/Figure/imputed_distributions2.jpg', dpi=500, bbox_inches='tight')
    
    kernel2.plot_mean_convergence(wspace=0.4, hspace=0.8)
    plt.savefig('./data/Figure/mean_convergence2.jpg', dpi=500, bbox_inches='tight')
    
    datasets = []
    res = []
    for i in range(kernel2.dataset_count()):
        datasets.append(kernel2.complete_data(i))
        dd = (datasets[i].mean() - data_missing.mean()) / data_missing.mean() * 100
        res.append(dd)
    
    mth_name = ["datasets0", "datasets1", "datasets2", "datasets3"]
    plt.style.use('default')
    plt.rcParams['font.family'] = ["Microsoft YaHei"]
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(26, 50))
    i = 0
    for j in range(len(datasets)):
        i += 1
        ax = plt.subplot(2, 2, i)
        ax = ((datasets[j].mean() - data_missing.mean()) / data_missing.mean() * 100).plot(
            kind="bar",
            title="多重插补结果：{}相对于原始数据集的变化量".format(mth_name[j])
        )
    ax.set_xlabel("", fontsize=10)
    ax.set_ylabel("change %")
    plt.savefig('./data/Figure/datasets_comparison.jpg', dpi=500, bbox_inches='tight')
    
    data_full = data_missing.copy()
    
    df.loc[:, "血红蛋白"] = datasets[0]["血红蛋白"]
    df.loc[:, "红细胞"] = datasets[0]["红细胞"]
    df.loc[:, "白蛋白"] = datasets[3]["白蛋白"]
    df.loc[:, "白细胞"] = datasets[0]["白细胞"]
    df.loc[:, "葡萄糖"] = datasets[2]["葡萄糖"]
    df.loc[:, "蛋白"] = datasets[2]["蛋白"]
    df.loc[:, "凝血酶原时间"] = datasets[2]["凝血酶原时间"]
    df.loc[:, "国际标准化的比值"] = datasets[2]["国际标准化的比值"]
    df.loc[:, "活化部分凝血活酶时间"] = datasets[2]["活化部分凝血活酶时间"]
    df.loc[:, "凝血酶时间"] = datasets[1]["凝血酶时间"]
    df.loc[:, "纤维蛋白原"] = datasets[2]["纤维蛋白原"]
    df.loc[:, "D-二聚体"] = datasets[0]["D-二聚体"]
    
    df.to_excel("./data/2020.7.11.xlsx", index=False)


if __name__ == "__main__":
    main()
