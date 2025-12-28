import sys

def error_message_detail(error,error_detail:sys):
    exc_type,exc_vale,exc_traceback=error_detail.exc_info()
    file_name=exc_traceback.tb_frame.f_code.co_filename
    error_message=(
        f"error occured in python script name {file_name}"
        f"line number {exc_traceback.tb_lineno}"
        f"error message {str(error)}"
    )

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)
    
    def __str__(self):
        return self.error_message