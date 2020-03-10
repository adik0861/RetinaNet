

from colors import color
import shutil



def banner(text):
    cols = shutil.get_terminal_size()[0]
    _pad = int((cols - len(text))/2)
    pad = _pad * ' '
    line = cols * '_'
    text = ''.join([pad, text, pad])
    text = '\n'.join([line, text])
    message = color(f'{text}', bg='yellow', style='bold')
    print(message)
    
banner('Training Epoch 4')