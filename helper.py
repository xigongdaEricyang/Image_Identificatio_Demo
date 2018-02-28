import sys
import numpy as np
import base64

def base64_encode_image(image):
    return base64.b64encode(image).decode("utf-8")

def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
	# serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding="utf-8")
 
	# convert the string to a NumPy array using the supplied data
	# type and target shape
	a = np.frombuffer(base64.decodestring(a), dtype=dtype)
	a = a.reshape(shape)
 
	# return the decoded image
	return a