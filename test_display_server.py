import sys
sys.path.append("X:\\")
sys.path.append("S:\\Schleier Lab Dropbox\\Cavity Lab Data\\Cavity Lab Scripts\\cavity_analysis")  # nopep8
import zprocess, time
port = 22552
host = '171.64.56.66'
file = 'S:/Schleier Lab Dropbox/Cavity Lab Data/2023/2023-01/2023-01-23/2023-01-16-PulseBlaster/0012_2023-01-16-PulseBlaster_8.h5'
zprocess.zmq_get_raw(port, host, file.encode('utf-8'))
zprocess.zmq_get_raw(port, host)
time.sleep(0.5)
zprocess.zmq_get_raw(port, host, 'done'.encode('utf-8'))
zprocess.zmq_get_raw(port, host)
