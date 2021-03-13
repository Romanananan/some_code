import pandas as pd

from .Checker import Checker
from .GiniVariablesChecker import GiniVariablesChecker
from .BinsMetricsChecker import BinsMetricsChecker
from .PSIVariablesChecker import PSIVariablesChecker
from .PermutationImportanceChecker import PermutationImportanceChecker
from .GiniStabilityChecker import GiniStabilityChecker
from .DataStatisticsChecker import DataStatisticsChecker
from .ShapFeatureImportanceChecker import ShapFeatureImportanceChecker
from .RegressionMetricsChecker import RegressionMetricsChecker
from .TreeCorrChecker import TreeCorrChecker
from .RECChecker import RECChecker
from .UpliftChecker import UpliftChecker

# Validation Check creator class
class CreateReport:
    """
    Класс создания конкретного экземпляра репортера в зависимости от параметра
    """

    def __init__(self,
                 writer: pd.ExcelWriter,
                 model_name: str,
                 model,
                 features_list: list,
                 cat_features: list,
                 drop_features: list,
                 model_type: str,
                 target_transformer,
                 current_path: str,
                 handbook=None):
        self.model_name = model_name
        self.model = model
        self.features_list = features_list
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.model_type = model_type
        self.target_transformer = target_transformer
        self.writer = writer
        self.current_path = current_path
        self.handbook = handbook

    def create_reporter(self, reporter_type: str) -> Checker:
        if reporter_type == "Gini_vars":
            return GiniVariablesChecker(writer=self.writer,
                                        model_name=self.model_name,
                                        model=self.model,
                                        features_list=self.features_list,
                                        cat_features=self.cat_features,
                                        drop_features=self.drop_features,
                                        handbook=self.handbook)

        elif reporter_type == "models_report":
            return BinsMetricsChecker(writer=self.writer,
                                      model_name=self.model_name,
                                      model=self.model,
                                      features_list=self.features_list,
                                      model_type=self.model_type,
                                      target_transformer=self.target_transformer,
                                      current_path=self.current_path)

        elif reporter_type == "PSI":
            return PSIVariablesChecker(writer=self.writer,
                                       model_name=self.model_name,
                                       model=self.model,
                                       features_list=self.features_list,
                                       cat_features=self.cat_features,
                                       drop_features=self.drop_features,
                                       model_type=self.model_type)

        elif reporter_type == "Permutation_imp":
            return PermutationImportanceChecker(writer=self.writer,
                                                model_name=self.model_name,
                                                model=self.model,
                                                features_list=self.features_list,
                                                cat_features=self.cat_features,
                                                drop_features=self.drop_features,
                                                model_type=self.model_type,
                                                current_path=self.current_path)

        elif reporter_type == "Gini_stability":
            return GiniStabilityChecker(writer=self.writer,
                                        model_name=self.model_name,
                                        model=self.model,
                                        features_list=self.features_list,
                                        cat_features=self.cat_features,
                                        drop_features=self.drop_features)

        elif reporter_type == "data_stats":
            return DataStatisticsChecker(
                writer=self.writer,
                model_name=self.model_name,
                model=self.model,
                features_list=self.features_list,
                cat_features=self.cat_features,
                drop_features=self.drop_features,
                model_type=self.model_type,
                target_transformer=self.target_transformer)

        elif reporter_type == "SHAP":
            return ShapFeatureImportanceChecker(writer=self.writer,
                                                model_name=self.model_name,
                                                model=self.model,
                                                features_list=self.features_list,
                                                cat_features=self.cat_features,
                                                drop_features=self.drop_features,
                                                current_path=self.current_path)
        elif reporter_type == "Uplift":
            return UpliftChecker(writer=self.writer,
                                 model_name=self.model_name,
                                 model=self.model,
                                 features_list=self.features_list,
                                 model_type=self.model_type,
                                 cat_features=self.cat_features,
                                 drop_features=self.drop_features,
                                 current_path=self.current_path)

        elif reporter_type == "RegMetrics":
            return RegressionMetricsChecker(
                writer=self.writer,
                model_name=self.model_name,
                model=self.model,
                features_list=self.features_list,
                cat_features=self.cat_features,
                drop_features=self.drop_features,
                target_transformer=self.target_transformer,
                current_path=self.current_path)

        elif reporter_type == "TreeCorr":
            return TreeCorrChecker(writer=self.writer,
                                   model_name=self.model_name,
                                   model=self.model,
                                   features_list=self.features_list,
                                   cat_features=self.cat_features,
                                   drop_features=self.drop_features)

        elif reporter_type == "REC":
            return RECChecker(writer=self.writer,
                              model_name=self.model_name,
                              model=self.model,
                              features_list=self.features_list,
                              cat_features=self.cat_features,
                              drop_features=self.drop_features,
                              target_transformation=self.target_transformer,
                              current_path=self.current_path)
