import sys
from src.logger import logging
def error_message_detail(error,error_detail:sys):
    # The third part gives detail about on which line the exception is there and 
    # what is the exception
    
    # the exc_info() returns a traceback object that holds references to all the variables of an exception
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured at line {exc_tb.tb_lineno} | File name  = {file_name} | Error Mesasge  = {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        # calls the constructor of the super class 
        # with some initialization error_message
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    # when we print : this method is called by default 
    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by 0 error")
        raise CustomException(e,sys)