import sys

class customexception(Exception): # for fetching details like file name, line number we are using this method
    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message
        # In which file,which line number and error message we are getting error 
        # In below line _,_, refers to 2 variables as outcome from exc_info which are not important and can be ignored
        _,_,exc_tb = error_details.exc_info() # exc_tb is execution traceback

        self.lineno = exc_tb.tb_lineno # getting line number
        self.file_name = exc_tb.tb_frame.f_code.co_filename # getting file name 

    def __str__(self): # for printing message we are using this method
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name,self.lineno,str(self.error_message)
        )
    