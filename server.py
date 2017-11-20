import io
import os
import tempfile

import tensorflow as tf
from PIL import Image
from base64 import b64decode
from flask import request
from flask_api import FlaskAPI

MAX_K = 5

TF_GRAPH = "{base_path}/inception_model/graph.pb".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
TF_LABELS = "{base_path}/inception_model/labels.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))


def load_graph():
    sess = tf.Session()
    with tf.gfile.FastGFile(TF_GRAPH, "rb") as tf_graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(tf_graph.read())
        _ = tf.import_graph_def(graph_def, name="")
    label_lines = [line.rstrip() for line in tf.gfile.GFile(TF_LABELS)]
    softmax_tensor = sess.graph.get_tensor_by_name("softmax:0")
    return sess, softmax_tensor, label_lines


#SESS, GRAPH_TENSOR, LABELS = load_graph()

app = FlaskAPI(__name__)

@app.route("/")
def ping():
    return { "success": True }

@app.route("/classify/", methods=["POST"])
def classifyImage():
  """
  Classify the image from image data.
  """
  data = { "success": False }
  image = request.data.get("image", None)
  temp = tempfile.NamedTemporaryFile()

  return { "success": True, "acne": 0.5, "skin cancer": 0.2 }

  if image is not None:
    plain_data = b64decode(image)
    temp.write(plain_data)

    classify_result = tf_classify(temp)
    temp.close()

    if classify_result:
      data.update({ "success": True })
      for res in classify_result:
        data[res[0]] = '{:f}'.format(res[1])

  return data

def tf_classify(image_file):
  result = list()

  image_data = tf.gfile.FastGFile(image_file.name, 'rb').read()

  predictions = SESS.run(GRAPH_TENSOR, {'DecodeJpeg/contents:0': image_data})
  predictions = predictions[0][:len(LABELS)]
  top_k = predictions.argsort()[-MAX_K:][::-1]
  for node_id in top_k:
      label_string = LABELS[node_id]
      score = predictions[node_id]
      result.append([label_string, score])

  return result

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=6000)