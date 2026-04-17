import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    mean_squared_error, r2_score, f1_score
)

class ModelAuditor:
    def __init__(self, task_type: str):
        self.task_type = task_type
        # Create a unique directory for this specific run for transparency
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"run_audit_{self.run_id}"
        os.makedirs(self.run_dir, exist_ok=True)

    def log_tournament_results(self, leaderboard_df: pd.DataFrame):
        """
        Saves the initial sprint results to a CSV and generates a comparison plot.
        Compatible with the output of ModelSelector.
        """

        csv_path = os.path.join(self.run_dir, "initial_tournament_leaderboard.csv")
        leaderboard_df.to_csv(csv_path, index=False)
        
        # Plotting the family performance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='score', y='family', data=leaderboard_df, palette='magma')
        plt.title(f"Tournament Results: {self.task_type.capitalize()}")
        plt.xlabel("Best Probe Score (Mean CV)")
        plt.ylabel("Algorithm Family")
        plt.tight_layout()
        
        plot_path = os.path.join(self.run_dir, "tournament_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Tournament audit saved to: {self.run_dir}")

    def perform_deep_audit(self, model, X_test, y_test, model_name: str):
        """
        Generates final performance metrics and plots for the top selected models.
        """
        y_pred = model.predict(X_test)
        results_file = os.path.join(self.run_dir, f"final_metrics_{model_name}.txt")

        if self.task_type == 'classification':
            # 1. Classification Report
            report = classification_report(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            with open(results_file, "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Weighted F1-Score: {f1:.4f}\n")
                f.write("-" * 30 + "\n")
                f.write(report)

            # 2. Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
            plt.title(f"Confusion Matrix - {model_name}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(os.path.join(self.run_dir, f"cm_{model_name}.png"))
            plt.close()

        else: # Regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            with open(results_file, "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"MSE: {mse:.4f}\n")
                f.write(f"R2 Score: {r2:.4f}\n")

            # Prediction Fit Plot
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.title(f"Actual vs Predicted - {model_name}")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.savefig(os.path.join(self.run_dir, f"regression_fit_{model_name}.png"))
            plt.close()

        print(f"📊 Deep audit for {model_name} completed.")