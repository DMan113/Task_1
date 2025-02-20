import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class ClassificationMetrics:
    """
    A utility class for evaluating classification performance using standard metrics.
    """
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate precision, recall, and F1-score for each class.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.

        Returns:
            dict: A dictionary containing per-class and average metrics, along with the confusion matrix.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

        per_class_metrics = []
        for i in range(10):
            per_class_metrics.append({
                'class': i,
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i]
            })

        avg_metrics = {
            'avg_precision': np.mean(precision),
            'avg_recall': np.mean(recall),
            'avg_f1': np.mean(f1)
        }

        return {
            'per_class_metrics': per_class_metrics,
            'average_metrics': avg_metrics,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    @staticmethod
    def print_metrics(metrics):
        """
        Print classification metrics in a readable format.

        Args:
            metrics (dict): A dictionary containing classification metrics.
        """
        print("\n=== Classification Metrics ===")
        print("\nPer-class metrics:")
        for class_metric in metrics['per_class_metrics']:
            print(f"\nClass {class_metric['class']}:")
            print(f"Precision: {class_metric['precision']:.4f}")
            print(f"Recall: {class_metric['recall']:.4f}")
            print(f"F1-score: {class_metric['f1_score']:.4f}")

        print("\nAverage metrics:")
        print(f"Average Precision: {metrics['average_metrics']['avg_precision']:.4f}")
        print(f"Average Recall: {metrics['average_metrics']['avg_recall']:.4f}")
        print(f"Average F1-score: {metrics['average_metrics']['avg_f1']:.4f}")
