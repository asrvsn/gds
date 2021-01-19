''' ZMQ utilities ''' 

import ujson
import zmq
import cloudpickle as pickle

zmq_pubsub_port = 8079

def ipc_tx():
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PUSH)
	sock.bind('ipc:///tmp/gns.ipc')
	tx = lambda msg: sock.send_json(msg)
	return ctx, tx

def ipc_rx():
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PULL)
	sock.connect('ipc:///tmp/gns.ipc')
	rx = lambda: sock.recv_json()
	return ctx, rx

def pubsub_tx():
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PUB)
	sock.bind('tcp://127.0.0.1:{}'.format(zmq_pubsub_port))
	tx = lambda msg: sock.send_multipart(ujson.dumps(mp))
	return ctx, tx

def pubsub_rx():
	ctx = zmq.Context()
	sock = ctx.socket(zmq.SUB)
	sock.connect('tcp://127.0.0.1:{}'.format(zmq_pubsub_port))
	sock.setsockopt(zmq.SUBSCRIBE, b'')
	rx = lambda: sock.recv_string()
	return ctx, rx

def wire_pickle(x):
	return pickle.dumps(x).decode('latin1')

def wire_unpickle(x):
	return pickle.loads(x.encode('latin1'))