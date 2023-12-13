from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio import session

put_info("Modify the code and click Run button to start.")

def f(x):
    print(x)
    return x
put_buttons([{"value": f'Accept', "label": f'Accept 123', "color": "info"}], onclick=lambda x: f(x))