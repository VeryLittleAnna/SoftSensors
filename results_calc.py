import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils import RMSELoss
from sklearn.metrics import r2_score
from collections import defaultdict
import plotly
from scipy import stats
import plotly.express as px
import statsmodels.api as sm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, hinge_loss
from scipy.stats import pearsonr, spearmanr
import plotly.offline as pyo
import pickle
import os

METRICS = ["corr", "pvalue", "r2", "rmse", "mae"]

rmse = RMSELoss()
def compute_metrics(y, y_pred, answer=None):
    if answer is None:
        answer = dict()
    answer['rmse'] = rmse(y_pred, y).item()
    answer['r2'] = r2_score(y, y_pred)
    answer['mae'] = np.mean(np.abs(y - y_pred))
    answer['corr'], answer['pvalue'] = pearsonr(y_pred, y)
    return answer

def reduce(metrics):
    answer = defaultdict(float)
    for m in metrics:
        for (key, value) in m.items():
            answer[key] += value
    for key in answer.keys():
        answer[key] /= len(metrics)
    return answer


class EvalModel:
    def __init__(self, plot=True, display=False, type_of_test="CV"):
        self.plot = plot 
        self.display = display 
        self.type_of_test = type_of_test
        self.methods = {"CV": self._calcCV, "fixed":self._calcFixed, "TimeSeries":self._calcTimeSeries}
        
    def _calcCV(self, results, name, metrics, y_pred_all, y_true_all, sizes):
        reduced_metrics = reduce(metrics)
        m = pd.concat([pd.DataFrame(reduced_metrics, index=["reduced"]), pd.DataFrame(metrics)])

        fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                            subplot_titles=("Прогноз", "Гетероскедастичность", "Q-Q график"))
        # fig.add_trace(go.Scatter(x=np.arange(y_pred_all.shape[0]), y=y_pred_all, mode='lines', name="LAB (pred)",
        #                      marker=dict(size=5, color="red")), row=1, col=1)
        xs = np.cumsum([0] + [len(x) for x  in y_pred_all])
        y_true_all = np.concatenate(y_true_all)
        fig.add_trace(go.Scatter(x=np.arange(y_true_all.shape[0]), y=y_true_all, mode='lines', name="true",
                                 marker=dict(size=5, color="green")), row=1, col=1)
        for i in range(len(y_pred_all)):
            fig.add_trace(go.Scatter(x=np.arange(xs[i], y_pred_all[i].shape[0] + xs[i]), y=y_pred_all[i], mode='lines', name="pred " + str(i),
                                marker=dict(size=5)), row=1, col=1)
        fig.update_xaxes(title_text="Время", row=1, col=1)
        fig.update_yaxes(title_text="Лабораторные показания", row=1, col=1)
        #
        y_pred_all = np.concatenate(y_pred_all)
        residual = y_true_all - y_pred_all
        fig.add_trace(go.Scatter(x=y_pred_all, y=residual.flatten(), mode='markers', 
                                 name="Residuals", showlegend=False), row=2, col=1)
        fig.add_hline(0, row=2, col=1, name=None)
        fig['layout']['xaxis2']['title']='Pred'
        fig['layout']['yaxis2']['title']='Residuals'
        # plot qq
        qq = stats.probplot(residual.flatten())
        x = np.array([qq[0][0][0], qq[0][0][-1]])
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name="Residuals", 
                                 showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines', showlegend=False), row=2, col=2)
        #fig.update_traces(marker={'size': 3})
        #names = {'Plot 1':'Прогноз', 'Plot 2':'Гетероскедастичность', 'Plot 3':'Q-Q график'}
        #fig.for_each_annotation(lambda a: a.update(text = names[a.text]))
        fig.update_layout(height=600, width=800, title_text=f"{name} {results['sensor']} {self.type_of_test}")
        
        self.display and fig.show()
        return m, fig

    def _calcFixed(self, results, name, metrics, y_pred_all, y_true_all, sizes):
        for i in range(len(metrics)):
            metrics[i]['train_size'] = sizes[i]
        reduced_metrics = reduce(metrics)
        m = pd.concat([pd.DataFrame(reduced_metrics, index=["reduced"]), pd.DataFrame(metrics)])

        fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                            subplot_titles=("Прогноз", "Зависимость от размера: RMSE, MAE", "Зависимость от размера: corr, r2"))
        # fig.add_trace(go.Scatter(x=np.arange(y_pred_all.shape[0]), y=y_pred_all, mode='lines', name="LAB (pred)",
        #                      marker=dict(size=5, color="red")), row=1, col=1)
        y_true_all = y_true_all[0]
        fig.add_trace(go.Scatter(x=np.arange(y_true_all.shape[0]), y=y_true_all, mode='lines', name="true",
                                 marker=dict(size=5, color="green")), row=1, col=1)
        for i in range(len(y_pred_all)):
            fig.add_trace(go.Scatter(x=np.arange(y_pred_all[i].shape[0]), y=y_pred_all[i], mode='lines', name="pred " + str(sizes[i]),
                                marker=dict(size=5)), row=1, col=1)
        fig.update_xaxes(title_text="Время", row=1, col=1)
        fig.update_yaxes(title_text="Лабораторные показания", row=1, col=1)
        fig.update_layout(height=600, width=800, title_text=f"{name} {results['sensor']} {self.type_of_test}")
        #
        fig.add_trace(go.Scatter(x=sizes, y=m['rmse'], mode='lines', name='rmse', marker=dict(size=5, color='orange')), row=2, col=1)  
        fig.add_trace(go.Scatter(x=sizes, y=m['mae'], mode='lines', name='mae', marker=dict(size=5, color='red')), row=2, col=1)  

        fig.add_trace(go.Scatter(x=sizes, y=m['r2'], mode='lines', name='r2', marker=dict(size=5, color='violet')), row=2, col=2)  
        fig.add_trace(go.Scatter(x=sizes, y=m['corr'], mode='lines', name='corr', marker=dict(size=5, color='gray')), row=2, col=2)  

        self.display and fig.show()
        return m, fig

    def _calcTimeSeries(self, results, name, metrics, y_pred_all, y_true_all, sizes):
        for i in range(len(metrics)):
            metrics[i]['train_size'] = sizes[i]
        reduced_metrics = reduce(metrics)
        m = pd.concat([pd.DataFrame(reduced_metrics, index=["reduced"]), pd.DataFrame(metrics)])

        fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                            subplot_titles=("Прогноз", "Зависимость от размера: RMSE, MAE", "Зависимость от размера: corr, r2"))
        y_true_all = np.concatenate(y_true_all)
        xs = np.cumsum([0] + [len(x) for x  in y_pred_all])

        fig.add_trace(go.Scatter(x=np.arange(y_true_all.shape[0]), y=y_true_all, mode='lines', name="true",
                                 marker=dict(size=5, color="green")), row=1, col=1)
        for i in range(len(y_pred_all)):
            fig.add_trace(go.Scatter(x=np.arange(xs[i], y_pred_all[i].shape[0] + xs[i]), y=y_pred_all[i], mode='lines', name="pred " + str(sizes[i]),
                                marker=dict(size=5)), row=1, col=1)
        fig.update_xaxes(title_text="Время", row=1, col=1)
        fig.update_yaxes(title_text="Лабораторные показания", row=1, col=1)
        fig.update_layout(height=600, width=800, title_text=f"{name} {results['sensor']} {self.type_of_test}")
        #
        fig.add_trace(go.Scatter(x=sizes, y=m['rmse'], mode='lines', name='rmse', marker=dict(size=5, color='orange')), row=2, col=1)  
        fig.add_trace(go.Scatter(x=sizes, y=m['mae'], mode='lines', name='mae', marker=dict(size=5, color='red')), row=2, col=1)  

        fig.add_trace(go.Scatter(x=sizes, y=m['r2'], mode='lines', name='r2', marker=dict(size=5, color='violet')), row=2, col=2)  
        fig.add_trace(go.Scatter(x=sizes, y=m['corr'], mode='lines', name='corr', marker=dict(size=5, color='gray')), row=2, col=2)  

        self.display and fig.show()
        return m, fig

    def __call__(self, results, name):
        folds = len(results["folds"])
        y_true_all, y_pred_all = [], []
        metrics = []
        sizes = []
        for fold in results['folds']:
            sizes.append((fold['mode'] == 0).sum())
            y_true = fold.loc[fold['mode'] == 2]['y'].to_numpy()
            y_pred = fold.loc[fold['mode'] == 2]['y_pred'].to_numpy()
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            metrics.append(compute_metrics(y_true, y_pred))
        m, fig = self.methods[self.type_of_test](results, name, metrics, y_pred_all, y_true_all, sizes)
        m.name = f"{name}_{results['sensor']}_{self.type_of_test}"
        return m, fig
    

def recalculate(folder=None, paths=None, plots_folder=None, type_of_test=None):
    if folder is not None:
        if isinstance(folder, list):
            paths = [(os.path.join(folder_, x), x) for folder_ in folder for x in os.listdir(folder_) if 'pickle' in x]
        else:
            paths = [(os.path.join(folder, x), x) for x in os.listdir(folder) if 'pickle' in x]
    cols = ["name", "type_of_test"] + METRICS + ["test_fold", "table_path", "plot_path", "params"]
    cnt = 0
    results_df = pd.DataFrame(columns=cols, index=range(len(paths)))
    for i, (path, filename) in enumerate(paths):
        with open(path, "rb") as f:
            file = pickle.load(f)
        if 'type_of_test' not in file:
            file['type_of_test'] = "CV"
        else:
            results_df.iloc[cnt]['test_fold'] = -1 #file['test_fold']
        if type_of_test is not None and file['type_of_test'] != type_of_test:
            continue
        plot_filename = os.path.join(plots_folder, f"{filename}.html")
        dashboard = open(plot_filename, 'w')
        dashboard.write("<html><head></head><body>" + "\n")
        # metrics, fig = eval_model(file, file['name'], plot=True, type_of_test=type_of_test)
        metrics, fig = EvalModel(plot=True, display=False, type_of_test=type_of_test)(file, file['name'])
        inner_html = fig.to_html(include_plotlyjs=True).split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
        dashboard.write(pd.DataFrame(metrics).to_html())
        # include_plotlyjs = False
        dashboard.write("</body></html>" + "\n")
        dashboard.close()
        results_df.iloc[cnt]['name'] = file['name']
   
        results_df.iloc[cnt]['type_of_test'] = file['type_of_test']
        if type_of_test == 'CV':
            results_df.iloc[cnt][METRICS] = metrics.iloc[0][METRICS].values
        results_df.iloc[cnt]['params'] = file['params']
        results_df.iloc[cnt]['table_path'] = path 
        results_df.iloc[cnt]['plot_path'] = plot_filename
        cnt += 1
    results_df.drop(axis=0, index=range(cnt, len(paths)), inplace=True)
    results_df.sort_values(by=['rmse'], ascending=True, axis=0, inplace=True)
    return results_df