import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score, f1_score


class ModelAuditor:
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.run_id= datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"run_audit_{self.run_id}"
        os.makedirs(self.run_dir, exist_ok=True)

    
    def log_tournament_results(self, leaderboard_df : pd.DataFrame):
        '''This function will save the initial sprint result into a csv and generate the comparison plot 
        create a csv_path from run_dir var in the self. use initial_tournament_loaderboard.csv as name and then write the leaderboard_df to csv file.
        Plot a barplot with x=score and y=family , data bein leaderboard_df, palette =magma  '''

        csv_path = os.path.join(self.run_dir, 'initial_tournament_leaderboard.csv')
        leaderboard_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x='score',y='family',data=leaderboard_df, palette = 'magma')
        plt.title(f'Tournament Result : {self.task_type.capitalize()}')
        plt.xlabel('Best probe score (mean CV)')
        plt.ylabel('Algorithm family')
        plt.tight_layout()


        plot_path= os.path.join(self.run_dir, 'tournament_plot.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"tournament result saved to {self.run_dir}")


    def perform_deep_audit(self, model, x_test, y_test, model_name:str):
        y_pred= model.predict(x_test)
        result_file= os.path.join(self.run_dir, f'final_metrics_{model_name}.txt')

        if self.task_type=='classification':
            report = classification_report(y_test, y_pred)
            f1= f1_score (y_test, y_pred, average='weighted')

            with open(result_file, 'w') as f:
                f.write(f"Model : {model_name}\n")
                f.write(f"Weighted F1 Score : {f1:.4f}\n")
                f.write('-'*30 + "\n")
                f.write(report)

            
            cm= confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, cmap='Purples',annot=True, fmt='d')
            plt.title(f"Confusion Matrix - {model_name}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(os.path.join(self.run_dir,f"cm_{model_name}.png"))
            plt.close()

        else:
            mse= mean_squared_error(y_test, y_pred)
            r2=r2_score(y_test, y_pred)

            with open(result_file, 'w') as f:
                f.write(f"Model - {model_name}")
                f.write(f"MSE - {mse:.4f}")
                f.write(f"r2_score - {r2:.4f}")
            
            plt.figure(figsize=(8,6))
            plt.scatter(y_test, y_pred, alpha=0.5,color='teal')
            plt.plot([y_test.min(), y_pred.min() ], [y_test.max(), y_pred.max()], 'r--' ,lw=2)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.savefig(os.path.join(self.run_dir, f"regression metrics {model_name}.png"))
            plt.close()

        
        print(f"Deep audit for model - {model_name} completed")



