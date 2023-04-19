import os
import sys
from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import  load_object, save_object, upload_file


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class CustomModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X):
        transformed_feature = self.preprocessing_object.transform(X)

        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = [
                        ('Linear Regression', LinearRegression()),
                        ('Ridge Regression', Ridge()),
                        ('Lasso Regression', Lasso()),
                        ('Random Forest Regression', RandomForestRegressor()),
                        ('Gradient Boosting Regression', GradientBoostingRegressor())
                    ]

            logging.info(f"Extracting model config file path")

            preprocessor = load_object(file_path=preprocessor_path)

            pipelines = []
            for name, model in models:
                pipelines.append((name, Pipeline([('preprocessor', preprocessor), (name, model)])))

           
           # Define the parameter grid for GridSearchCV
            param_grid = {
                'Linear Regression': {},
                'Ridge Regression': {'alpha': [0.01, 0.1, 1, 10]},
                'Lasso Regression': {'alpha': [0.01, 0.1, 1, 10]},
                'Random Forest Regression': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
                'Gradient Boosting Regression': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10], 'learning_rate': [0.01, 0.1, 1]},
                'Neural Network Regression': {'hidden_layer_sizes': [(50, 50), (100, 50, 25)], 'alpha': [0.0001, 0.001, 0.01]}
            }

            # Define the GridSearchCV object
            grid = GridSearchCV(estimator=pipelines, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

            # Fit the grid search object to the training data
            grid.fit(x_train, y_train)

            logging.info(f"Best found model on both training and testing dataset")

            best_model = grid.best_estimator_

            

            custom_model = CustomModel(
                preprocessing_object=preprocessor,
                trained_model_object=best_model,
            )

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_file_path}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=custom_model,
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)

            upload_file(
                from_filename=self.model_trainer_config.trained_model_file_path,
                to_filename="model.pkl",
                bucket_name="ineuron-test-bucket-123",
            )

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
