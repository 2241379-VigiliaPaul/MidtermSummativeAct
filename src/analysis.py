# Analysis Script - Compare model performance across all four models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

def load_evaluation_results():
    # Define model names and their result files
    models = ['decision_tree', 'naive_bayes', 'random_forest', 'svm']
    results = {}
    
    for model in models:
        # Try to load saved results if they exist
        result_file = f'results/{model}_evaluation.txt'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[model] = f.read()
    
    return results

def compare_test_vs_validation():
    # Load evaluation results
    results = load_evaluation_results()
    
    if not results:
        print("No evaluation results found. Run model files first.")
        return
    
    print("=== MODEL COMPARISON: TEST SET vs UNSEEN DATA ===\n")
    
    # Parse results and extract key metrics
    model_metrics = {}
    for model_name, result_text in results.items():
        if result_text:
            lines = result_text.split('\n')
            metrics = {}
            for line in lines:
                if ':' in line and not line.startswith('---'):
                    key, value = line.split(':', 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        continue
            model_metrics[model_name] = metrics
    
    # Create comparison table
    comparison_data = []
    for model_name, metrics in model_metrics.items():
        if metrics:
            row = {'Model': model_name.replace('_', ' ').title()}
            # Add test set metrics
            if 'Accuracy' in metrics:
                row['Test_Accuracy'] = metrics['Accuracy']
            if 'Precision' in metrics:
                row['Test_Precision'] = metrics['Precision']
            if 'Recall' in metrics:
                row['Test_Recall'] = metrics['Recall']
            if 'F1-score' in metrics:
                row['Test_F1'] = metrics['F1-score']
            if 'Specificity' in metrics:
                row['Test_Specificity'] = metrics['Specificity']
            if 'ROC-AUC' in metrics:
                row['Test_ROC-AUC'] = metrics['ROC-AUC']
            comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
    
    # Calculate performance differences
    print("\n=== PERFORMANCE DIFFERENCES ===")
    for model_name in model_metrics:
        metrics = model_metrics[model_name]
        if metrics and 'Accuracy' in metrics:
            print(f"{model_name.replace('_', ' ').title()}: Test Accuracy = {metrics['Accuracy']:.4f}")

def generate_comparison_report():
    # Load and compare results
    compare_test_vs_validation()
    
    # Create performance ranking
    results = load_evaluation_results()
    if not results:
        return
    
    print("\n=== MODEL RANKINGS ===")
    
    # Parse metrics for ranking
    model_scores = {}
    for model_name, result_text in results.items():
        if result_text:
            lines = result_text.split('\n')
            for line in lines:
                if 'Accuracy:' in line:
                    try:
                        accuracy = float(line.split(':')[1].strip())
                        model_scores[model_name] = accuracy
                        break
                    except (ValueError, IndexError):
                        continue
    
    # Sort models by accuracy
    if model_scores:
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("Models ranked by Test Accuracy:")
        for i, (model, score) in enumerate(ranked_models, 1):
            print(f"{i}. {model.replace('_', ' ').title()}: {score:.4f}")
    
    # Generate visualization
    try:
        if model_scores:
            models = list(model_scores.keys())
            scores = list(model_scores.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, [m.replace('_', ' ').title() for m in models], color=['skyblue', 'lightgreen', 'salmon', 'orange'])
            plt.title('Model Performance Comparison (Test Accuracy)', fontsize=14, fontweight='bold')
            plt.xlabel('Models', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{scores[i]:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save visualization
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
            print("\nVisualization saved to: results/model_comparison.png")
            
    except Exception as e:
        print(f"Error creating visualization: {e}")

def analyze_model_differences():
    results = load_evaluation_results()
    if not results:
        return
    
    print("\n=== STATISTICAL ANALYSIS ===")
    
    # Extract accuracy scores for analysis
    accuracies = []
    model_names = []
    
    for model_name, result_text in results.items():
        if result_text:
            lines = result_text.split('\n')
            for line in lines:
                if 'Accuracy:' in line:
                    try:
                        accuracy = float(line.split(':')[1].strip())
                        accuracies.append(accuracy)
                        model_names.append(model_name.replace('_', ' ').title())
                        break
                    except (ValueError, IndexError):
                        continue
    
    if len(accuracies) >= 2:
        # Calculate statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"Mean Accuracy: {mean_acc:.4f}")
        print(f"Standard Deviation: {std_acc:.4f}")
        print(f"Accuracy Range: {min(accuracies):.4f} - {max(accuracies):.4f}")
        
        # Identify best and worst performing models
        best_idx = np.argmax(accuracies)
        worst_idx = np.argmin(accuracies)
        
        print(f"\nBest Model: {model_names[best_idx]} ({accuracies[best_idx]:.4f})")
        print(f"Worst Model: {model_names[worst_idx]} ({accuracies[worst_idx]:.4f})")

if __name__ == "__main__":
    print("Starting comprehensive model analysis...")
    
    # Load all evaluation results
    results = load_evaluation_results()
    
    if not results:
        print("No evaluation results found. Please run model files first.")
        print("Expected files in results/ directory:")
        print("- decision_tree_evaluation.txt")
        print("- naive_bayes_evaluation.txt") 
        print("- random_forest_evaluation.txt")
        print("- svm_evaluation.txt")
    else:
        # Generate comprehensive comparison
        generate_comparison_report()
        
        # Analyze performance differences
        analyze_model_differences()
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Results saved to results/model_comparison.png")