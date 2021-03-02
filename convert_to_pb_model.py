import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# import tensorflow.python.keras.backend as K
import numpy as np

# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     """
#     Freezes the state of a session into a pruned computation graph.

#     Creates a new computation graph where variable nodes are replaced by
#     constants taking their current value in the session. The new graph will be
#     pruned so subgraphs that are not necessary to compute the requested
#     outputs are removed.
#     @param session The TensorFlow session to be frozen.
#     @param keep_var_names A list of variable names that should not be frozen,
#                           or None to freeze all the variables in the graph.
#     @param output_names Names of the relevant graph outputs.
#     @param clear_devices Remove the device directives from the graph for better portability.
#     @return The frozen graph definition.
#     """
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = tf.graph_util.convert_variables_to_constants(
#             session, input_graph_def, output_names, freeze_var_names)
#         return frozen_graph

# # SAVE KERAS MODEL AS TENSORFLOW(PB) MODEL
# wkdir = '/home/varat/Emotix/myGITHUB/Image-Classification'
# frozen_out_path = 'models'
# pb_filename = 'frozen_graph_pet.pb'
# # Load keras model
# model = tf.keras.models.load_model('models/saved_model_pet')
# frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(),
#     output_names=[out.op.name for out in model.outputs])
# tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)





# frozen_out_path = 'models'
# frozen_graph_filename = 'frozen_graph_pet'
# # Load keras model
# model = tf.keras.models.load_model('models/myModel_pet.h5')
# model.summary()

# # Convert Keras model to ConcreteFunction
# full_model = tf.function(lambda x: model(x))
# full_model = full_model.get_concrete_function(
#     tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# # Get frozen ConcreteFunction
# frozen_func = convert_variables_to_constants_v2(full_model)
# frozen_func.graph.as_graph_def()

# layers = [op.name for op in frozen_func.graph.get_operations()]
# print("-" * 60)
# print("Frozen model layers: ")
# for layer in layers:
#     print(layer)

# print("-" * 60)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)

# # Save frozen graph to disk
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir=frozen_out_path,
#                   name=f"{frozen_graph_filename}.pb",
#                   as_text=False)

# # Save its text representation
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir=frozen_out_path,
#                   name=f"{frozen_graph_filename}.pbtxt",
#                   as_text=True)