import tensorflow as tf

cluster_specification = {
    "ps": ["localhost:2222"], # list of parameter servers,
    "worker": ["localhost:2223", "localhost:2224"] # list of workers
}

def start_parameter_server(task_index, cluster_specification):
    cluster_spec = tf.train.ClusterSpec(cluster_specification)
    server = tf.train.Server(cluster_spec, job_name='ps', task_index=task_index)
    server.join()


if __name__ == '__main__':
    start_parameter_server(0, cluster_specification)