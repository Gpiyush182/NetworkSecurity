import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.main_utils.utils import load_object, save_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import mlflow
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self,best_model,clasificationmetric):
        with mlflow.start_run():
            f1_score = clasificationmetric.f1_score
            precision_score = clasificationmetric.precision_score
            recall_score = clasificationmetric.recall_score
            
            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
            
        
    def train_model(self,x_train,y_train,x_test,y_test):
        models = {"Logistic Regression": LogisticRegression(verbose=1),
                  "KNeighbors Classifier": KNeighborsClassifier(),
                  "Decision Tree Classifier": DecisionTreeClassifier(),
                  "Random Forest Classifier": RandomForestClassifier(verbose=1),
                  "AdaBoost Classifier": AdaBoostClassifier(),
                  "Gradient Boosting Classifier": GradientBoostingClassifier(verbose=1)}
        param = {
            "Decision Tree Classifier": {
                "criterion": ['gini', 'entropy', 'log_loss'],
                "splitter": ['best', 'random'],
                "max_features": ['sqrt', 'log2'],
            },
            "Random Forest Classifier": {
                "criterion": ['gini', 'entropy', 'log_loss'],
                "max_features": ['sqrt', 'log2', 'None'],
                "n_estimators":[2,4,8,46,59,96,256,128]
            },
            "AdaBoost Classifier": {
                "learning_rate": [0.1,0.01,1.0,0.5,0.8],
                "n_estimators": [1,3,5,7,9,39,79,307,453],
            },
            "Gradient Boosting Classifier": {
                "loss": ["log_loss","exponential"],
                "learning_rate": [0.1,0.01,1.0,0.5,0.8],
                "n_estimators": [1,3,8,46,59,96,256,307,453],
                "criterion": ['friedman_mse','squared_error'],
            },
            "Logistic Regression": {
                "penalty": ['l1','l2','elasticnet','None'],
                "multi_class": ['auto','ovr','multinomial'],
                "solver": ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'],
            },
            "KNeighbors Classifier": {
                "n_neighbors": [1,2,3,4,5,6,7,8,9],
                "weights": ['uniform','distance'],
                "algorithm": ['auto','ball_tree','kd_tree','brute'],   
            }
        }
        
        model_report: dict = evaluate_models(X_train = x_train,y_train = y_train, X_test = x_test, y_test=y_test, models=models,param=param )
        
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(x_train)
        
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        ## Track the mlflow
        self.track_mlflow(best_model,classification_train_metric)
        
        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model,classification_test_metric)
        
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        
        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
        
        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact
        
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            ## Loading training and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1], train_arr[:,-1],
                test_arr[:,:-1], test_arr[:,-1],
            )
            
            model_trainer_artifact = self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)