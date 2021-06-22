"""
Configuration

"""

class Configuration:

    # data parameters
    PRICE_COL   = 'Close'
    DATE_COL    = 'Date'

    # Model parameters
    OUTPUT_SIZE = 3
    INPUT_SIZE  = 10

    TRAIN_SIZE  = 0.7
    VAL_SIZE    = 0.2
    TEST_SIZE   = 0.1
