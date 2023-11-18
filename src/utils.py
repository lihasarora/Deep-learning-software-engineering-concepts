import os
import dill
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        # make directory if doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_model(X_train,y_train,X_test ,y_test ,models ):
    
    try:
        report  = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train) # train the model
            print("model training finished")
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)
            print("prediction finished")
            # model evaluation
            train_model_score = r2_score(y_train, y_train_pred)            
            test_model_score  = r2_score(y_test, y_test_pred)
            # store it in a dictionary
            report[list(models.keys())[i]] = test_model_score    
        return report
    except Exception as e:
        raise CustomException(e,sys)

if __name__ =='__main__':
    a = 6
    save_object("test/",a)