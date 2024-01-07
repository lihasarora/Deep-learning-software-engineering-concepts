import logging
import os
from datetime import datetime

# define the log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Create a logs_filename in the current working directory to get the logs noted down
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Now, we want to configure the logging module to write logs to the config file
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force = True
)

# __name__ is a special variable that is used to identify if the program
# is running as the main program or not. If this program is being executed
# as main program, the __name__ is set to __main__ by the compiler automatically

# If this program is imported as a module by some other program, it the 
# below code doesn't get executedd

if __name__ == "__main__":
    logging.info("Logging has started")