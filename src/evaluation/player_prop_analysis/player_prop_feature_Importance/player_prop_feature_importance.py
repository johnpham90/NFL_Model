"""
Enhanced feature importance analysis with RF/XGB support and correlation analysis.

Key additions:
- Support for both RandomForest and XGBoost models
- Correlation analysis to identify redundant features
- Feature selection recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle


class FeatureAnalyzer:
    """Analyze feature importance from trained player prop models."""
    
    def __init__(self, model_path: str):
        """Load model artifact."""
        self.model_path = Path(model_path)
        
        with open(self.model_path, 'rb') as f:
            self.artifact = pickle.load(f)
        
        self.model = self.artifact.model
        self.features = self.artifact.feature_columns
        self.model_name = self.model_path.stem
        self.model_type = type(self.model).__name__
    
    def get_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Extract feature importance from XGBoost or RandomForest.
        
        Args:
            importance_type: For XGBoost: 'gain', 'weight', or 'cover'
                           For RandomForest: ignored (uses feature_importances_)
        
        Returns:
            DataFrame with feature, importance, rank
        """
        # Detect model type and extract importance
        if hasattr(self.model, 'get_booster'):
            # XGBoost model
            importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
            df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in importance_dict.items()
            ])
        elif hasattr(self.model, 'feature_importances_'):
            # RandomForest or any sklearn model with feature_importances_
            df = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            })
        else:
            raise ValueError(f"Model type {self.model_type} not supported for importance extraction")
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        df['importance_pct'] = (df['importance'] / df['importance'].sum() * 100).round(2)
        df['cumulative_pct'] = df['importance_pct'].cumsum().round(2)
        
        return df
    
    def categorize(self, feature_name: str) -> str:
        """Categorize feature by type."""
        f = feature_name.lower()
        
        if 'opp_def_' in f or '_allowed' in f:
            return 'defense'
        elif any(x in f for x in ['snap', 'is_starter', 'touches', 'target_share']):
            return 'usage'
        elif any(x in f for x in ['spread', 'total', 'moneyline', 'implied', 'favorite']):
            return 'lines'
        elif '__player_roll' in f or '__player_std' in f:
            return 'rolling'
        elif '__player_hist' in f or '__hist' in f:
            return 'historical'
        elif f.startswith('has_') or f.startswith('is_'):
            return 'indicators'
        elif 'vs_opp' in f or 'matchup' in f:
            return 'matchup'
        elif any(x in f for x in ['_z', 'zscore', 'z_score']):
            return 'zscore'
        else:
            return 'other'
    
    def add_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add category column."""
        df = df.copy()
        df['category'] = df['feature'].apply(self.categorize)
        return df
    
    def summarize_by_category(self) -> pd.DataFrame:
        """Aggregate by category."""
        df = self.get_importance()
        df = self.add_categories(df)
        
        summary = df.groupby('category').agg({
            'importance': ['sum', 'mean', 'count'],
            'importance_pct': 'sum',
            'rank': 'min'
        }).round(2)
        
        summary.columns = ['total_importance', 'avg_importance', 'count', 'total_pct', 'best_rank']
        return summary.sort_values('total_importance', ascending=False)
    
    def get_correlation_matrix(self, X_train: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """
        Calculate feature correlation matrix.
        
        Args:
            X_train: Training data with all features
            threshold: Correlation threshold for flagging redundancy
            
        Returns:
            Correlation matrix DataFrame
        """
        # Only use features that are in the model
        available_features = [f for f in self.features if f in X_train.columns]
        corr_matrix = X_train[available_features].corr()
        return corr_matrix
    
    def find_redundant_features(
        self, 
        X_train: pd.DataFrame, 
        threshold: float = 0.90
    ) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated feature pairs (potential redundancy).
        
        Args:
            X_train: Training data with all features
            threshold: Correlation threshold (default 0.90)
            
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        corr_matrix = self.get_correlation_matrix(X_train)
        
        # Find pairs with correlation above threshold
        redundant = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    redundant.append((feat1, feat2, corr_val))
        
        # Sort by correlation magnitude
        redundant.sort(key=lambda x: abs(x[2]), reverse=True)
        return redundant
    
    def recommend_features_to_remove(
        self,
        X_train: pd.DataFrame,
        threshold: float = 0.90
    ) -> Dict[str, List[str]]:
        """
        Recommend which features to remove based on correlation and importance.
        
        For each redundant pair, recommends removing the less important feature.
        
        Args:
            X_train: Training data
            threshold: Correlation threshold
            
        Returns:
            Dict with 'keep' and 'remove' lists
        """
        redundant_pairs = self.find_redundant_features(X_train, threshold)
        importance = self.get_importance()
        importance_dict = dict(zip(importance['feature'], importance['importance']))
        
        keep = set(self.features)
        remove = set()
        
        for feat1, feat2, corr in redundant_pairs:
            # If both still in keep set
            if feat1 in keep and feat2 in keep:
                # Remove the less important one
                imp1 = importance_dict.get(feat1, 0)
                imp2 = importance_dict.get(feat2, 0)
                
                if imp1 < imp2:
                    remove.add(feat1)
                    keep.discard(feat1)
                else:
                    remove.add(feat2)
                    keep.discard(feat2)
        
        return {
            'keep': sorted(list(keep)),
            'remove': sorted(list(remove)),
            'removed_count': len(remove),
            'kept_count': len(keep)
        }
    
    def select_top_features(
        self, 
        n: Optional[int] = None, 
        cumulative_pct: Optional[float] = None
    ) -> List[str]:
        """
        Select top N features or features explaining X% of importance.
        
        Args:
            n: Number of top features to keep (e.g., 50)
            cumulative_pct: Cumulative importance % threshold (e.g., 90)
            
        Returns:
            List of selected feature names
        """
        df = self.get_importance()
        
        if n is not None:
            return df.head(n)['feature'].tolist()
        elif cumulative_pct is not None:
            return df[df['cumulative_pct'] <= cumulative_pct]['feature'].tolist()
        else:
            raise ValueError("Must provide either n or cumulative_pct")
    
    def print_summary(self, top_n: int = 15):
        """Print summary to console."""
        df = self.get_importance()
        df = self.add_categories(df)
        categories = self.summarize_by_category()
        
        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE: {self.model_name} ({self.model_type})")
        print(f"{'='*80}\n")
        
        print(f"TOP {top_n} FEATURES:")
        print("-" * 80)
        display_cols = ['rank', 'feature', 'importance', 'importance_pct', 'cumulative_pct', 'category']
        print(df[display_cols].head(top_n).to_string(index=False))
        
        print(f"\n{'='*80}")
        print("IMPORTANCE BY CATEGORY:")
        print("-" * 80)
        print(categories.to_string())
        
        # Feature selection suggestions
        top_50 = self.select_top_features(n=50)
        top_90pct = self.select_top_features(cumulative_pct=90)
        
        print(f"\n{'='*80}")
        print("FEATURE SELECTION RECOMMENDATIONS:")
        print("-" * 80)
        print(f"Top 50 features explain: {df.head(50)['cumulative_pct'].iloc[-1]:.1f}% of importance")
        print(f"90% importance threshold: {len(top_90pct)} features")
        print(f"Total features: {len(self.features)}")
        print(f"\n{'='*80}\n")
    
    def print_correlation_summary(
        self,
        X_train: pd.DataFrame,
        threshold: float = 0.90,
        top_n: int = 10
    ):
        """Print correlation analysis summary."""
        redundant = self.find_redundant_features(X_train, threshold)
        recommendations = self.recommend_features_to_remove(X_train, threshold)
        
        print(f"\n{'='*80}")
        print(f"CORRELATION ANALYSIS (threshold={threshold})")
        print(f"{'='*80}\n")
        
        print(f"Found {len(redundant)} redundant feature pairs")
        print(f"Recommend removing {recommendations['removed_count']} features")
        print(f"Would keep {recommendations['kept_count']} features")
        
        if redundant:
            print(f"\nTOP {min(top_n, len(redundant))} REDUNDANT PAIRS:")
            print("-" * 80)
            for feat1, feat2, corr in redundant[:top_n]:
                print(f"{corr:6.3f}  {feat1} <-> {feat2}")
        
        if recommendations['remove']:
            print(f"\nRECOMMENDED TO REMOVE ({len(recommendations['remove'])} features):")
            print("-" * 80)
            for feat in recommendations['remove'][:20]:
                print(f"  - {feat}")
            if len(recommendations['remove']) > 20:
                print(f"  ... and {len(recommendations['remove']) - 20} more")
        
        print(f"\n{'='*80}\n")
    
    def plot_top_features(self, n: int = 20, save_path: Optional[str] = None):
        """Bar chart of top features."""
        df = self.get_importance()
        df = self.add_categories(df)
        df = df.head(n)
        
        # Colors by category
        categories = df['category'].unique()
        colors = sns.color_palette('husl', n_colors=len(categories))
        color_map = dict(zip(categories, colors))
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(df)), df['importance'], 
                color=[color_map[c] for c in df['category']])
        
        plt.yticks(range(len(df)), df['feature'], fontsize=9)
        plt.xlabel('Importance', fontsize=11)
        plt.title(f'Top {n} Features - {self.model_name} ({self.model_type})', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Legend
        from matplotlib.patches import Patch
        legend = [Patch(facecolor=color_map[c], label=c.title()) for c in categories]
        plt.legend(handles=legend, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_categories(self, save_path: Optional[str] = None):
        """Bar chart of categories."""
        summary = self.summarize_by_category()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        summary['total_importance'].plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_xlabel('Total Importance', fontsize=11)
        ax1.set_title('Importance by Category', fontsize=12)
        
        summary['count'].plot(kind='barh', ax=ax2, color='coral')
        ax2.set_xlabel('Number of Features', fontsize=11)
        ax2.set_title('Features by Category', fontsize=12)
        
        plt.suptitle(f'{self.model_name} ({self.model_type})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_heatmap(
        self,
        X_train: pd.DataFrame,
        top_n: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Plot correlation heatmap for top N features.
        
        Args:
            X_train: Training data
            top_n: Number of top features to include
            save_path: Path to save figure
        """
        # Get top N features by importance
        top_features = self.select_top_features(n=top_n)
        available = [f for f in top_features if f in X_train.columns]
        
        if len(available) < 2:
            print("Not enough features available for correlation heatmap")
            return
        
        # Calculate correlation
        corr_matrix = X_train[available].corr()
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            cmap='RdBu_r', 
            center=0,
            vmin=-1, 
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=False
        )
        plt.title(f'Feature Correlation - Top {len(available)} Features\n{self.model_name}', 
                 fontsize=12, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_cumulative_importance(self, save_path: Optional[str] = None):
        """Plot cumulative importance curve."""
        df = self.get_importance()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(df)+1), df['cumulative_pct'], linewidth=2, color='steelblue')
        plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
        plt.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        
        plt.xlabel('Number of Features', fontsize=11)
        plt.ylabel('Cumulative Importance (%)', fontsize=11)
        plt.title(f'Cumulative Feature Importance - {self.model_name}', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def save_report(
        self,
        X_train: Optional[pd.DataFrame] = None,
        output_dir: str = 'artifacts/analysis',
        correlation_threshold: float = 0.90
    ):
        """
        Generate complete analysis report.
        
        Args:
            X_train: Training data for correlation analysis (optional)
            output_dir: Output directory
            correlation_threshold: Threshold for redundancy detection
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"GENERATING REPORT: {self.model_name}")
        print(f"{'='*80}\n")
        
        # Get data
        all_features = self.get_importance()
        all_features = self.add_categories(all_features)
        categories = self.summarize_by_category()
        
        # Save CSVs
        all_features.to_csv(output_path / f'{self.model_name}_features.csv', index=False)
        categories.to_csv(output_path / f'{self.model_name}_categories.csv')
        
        # Save plots
        self.plot_top_features(25, output_path / f'{self.model_name}_top25.png')
        self.plot_categories(output_path / f'{self.model_name}_categories.png')
        self.plot_cumulative_importance(output_path / f'{self.model_name}_cumulative.png')
        
        # Correlation analysis if training data provided
        if X_train is not None:
            print("Running correlation analysis...")
            redundant = self.find_redundant_features(X_train, correlation_threshold)
            recommendations = self.recommend_features_to_remove(X_train, correlation_threshold)
            
            # Save correlation data
            if redundant:
                redundant_df = pd.DataFrame(redundant, columns=['feature1', 'feature2', 'correlation'])
                redundant_df.to_csv(output_path / f'{self.model_name}_redundant_pairs.csv', index=False)
            
            pd.DataFrame({
                'feature': recommendations['remove']
            }).to_csv(output_path / f'{self.model_name}_recommend_remove.csv', index=False)
            
            # Save correlation plots
            self.plot_correlation_heatmap(X_train, 30, output_path / f'{self.model_name}_correlation.png')
            
            # Print correlation summary
            self.print_correlation_summary(X_train, correlation_threshold)
        
        print(f"\n✅ REPORT SAVED TO: {output_path}/")
        print(f"   Files generated:")
        print(f"   - {self.model_name}_features.csv ({len(all_features)} features)")
        print(f"   - {self.model_name}_categories.csv")
        print(f"   - {self.model_name}_top25.png")
        print(f"   - {self.model_name}_categories.png")
        print(f"   - {self.model_name}_cumulative.png")
        if X_train is not None:
            print(f"   - {self.model_name}_correlation.png")
            print(f"   - {self.model_name}_redundant_pairs.csv")
            print(f"   - {self.model_name}_recommend_remove.csv")
        
        # Print console summary
        self.print_summary(15)


def analyze(
    model_path: str, 
    X_train: Optional[pd.DataFrame] = None,
    output_dir: str = 'artifacts/analysis',
    correlation_threshold: float = 0.90
):
    """
    Quick analysis helper.
    
    Args:
        model_path: Path to saved model artifact
        X_train: Training data for correlation analysis (optional)
        output_dir: Output directory
        correlation_threshold: Threshold for redundancy detection
        
    Returns:
        FeatureAnalyzer instance
    """
    analyzer = FeatureAnalyzer(model_path)
    analyzer.save_report(X_train, output_dir, correlation_threshold)
    return analyzer


# For easy importing
__all__ = ['FeatureAnalyzer', 'analyze']