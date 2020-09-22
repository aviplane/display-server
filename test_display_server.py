import zprocess, time
port = 22552
host = '171.64.56.66'
file = 'X:\\2019\\2019-03\\2019-03-11\\2019-03-01-PairCreation\\20190311T112512_2019-03-01-PairCreation_0.h5'
zprocess.zmq_get_raw(port, host, file.encode('utf-8'))
zprocess.zmq_get_raw(port, host)
time.sleep(0.5)
zprocess.zmq_get_raw(port, host, 'done'.encode('utf-8'))
zprocess.zmq_get_raw(port, host)
