
from pywebio.output import *
from pywebio.input import * 
import time

put_markdown('## Hello there')

# put_text("I hope you are having a great day! Here is our menu")
#
# put_table([
#             ['Type', 'Content'],
#             ['html', put_html('X<sup>2</sup>')],
#             ['text', '<hr/>'],
#             ['buttons', put_buttons(['A', 'B'], onclick=...)],  # ..doc-only
#             ['buttons', put_buttons(['A', 'B'], onclick=put_text)],  # ..demo-only
#             ['markdown', put_markdown('`Awesome PyWebIO!`')],
#             ['file', put_file('hello.text', b'hello world')],
#             ['table', put_table([['A', 'B'], ['C', 'D']])]
#         ])
#
# with popup("Subscribe to the page"):
#     put_text("Join other foodies!")
#
# food = select("Choose your favorite food", ['noodle', 'chicken and rice'])
#
# put_text(f"You chose {food}. Please wait until it is served!")
#
# put_processbar('bar')
# for i in range(1, 11):
#     set_processbar('bar', i / 10)
#     time.sleep(0.1)
# put_markdown("Here is your food! Enjoy!")
#
# if food == 'noodle':
#     put_image(open('noodle.jpeg', 'rb').read())
# else:
#     put_image(open('chicken_and_rice.jpeg', 'rb').read())


num = 10
accept_i = []
def Onclick(x):
    global accept_i
    accept_i.append(x.split()[-1])
    return x




def Apply(x):
    global accept_i, L
    print(accept_i)
    accept_i = []
    L = []
    for i in range(num):
        L.append(["text", f"This is number {i}"])
        L.append(['buttons', put_buttons([f'Accept {i}', f'Reject {i}'], onclick=Onclick)])
    L.append(["text", f"DMMMMMMM {accept_i}"])
    put_table(L)
    return x
put_buttons(["APPLY"], onclick=Apply)
# put_file("You can download the food here", b"Hello")


